from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection
from langchain.tools import tool


# 1. 数据模型（保持不变）
class TextWithMeta(BaseModel):
    """单条文本的数据模型（含内容、多标签、元信息）"""
    text_id: str  # 文本唯一标识（方便后续更新/删除）
    content: str  # 核心文本内容（用于生成向量）

    # 标签，表示这个content是什么类型的信息
    states: bool=False
    capabilities: bool=False
    device_id_clues: bool=False
    usage_habits: bool=False
    others: bool=False

    # 修正1：create_time 用Field设置实时默认值（lambda确保实例化时实时计算）
    create_time: datetime = Field(default_factory=lambda: datetime.now(), description="创建时间")
    update_time: Optional[datetime] = None  # 更新时间（元信息，可选）
    source:  Optional[str] = None
    other_meta: Optional[dict] = None  # 其他自定义元信息（如作者、来源等）

class VectorDB():
    def __init__(self):
        # 初始化文本嵌入函数（保持不变）
        self.embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 轻量高效，支持中英文
        )
        # 初始化Chroma向量数据库（保持不变，支持持久化）
        self.client = chromadb.PersistentClient(path="./chroma_text_db")
        # 定义极小值，避免除零错误（保证d>0）
        self.epsilon = 1e-6
        # 定义默认距离（无匹配/空集合时使用，代表低匹配度）
        self.default_distance = 1.0

    def get_or_create_collection(self, collection_name: str, device_name: str="N/A") -> Collection:
        """
        获取或者创建以 "设备ID" 为名的集合
        :param collection_name: 设备ID（集合唯一标识，必传）
        :param device_name: 设备名称（可选，默认值为「N/A」，表示未知设备名称）
        :return: ChromaDB 集合对象
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={
                "description": "存储设备信息，关联多标签和创建/更新时间等元信息",
                "device_name": device_name}
        )

    def add_text_to_vector_db(self,text_data: TextWithMeta, collection: Collection):
        """将单条文本（含标签、元信息）存入向量数据库，tags列表拆分为独立字段"""
        # 1. 处理自定义元信息，避免内部包含None值
        other_meta = text_data.other_meta or {}
        cleaned_other_meta = {k: v if v is not None else "N/A" for k, v in other_meta.items()}

        # 2. 初始化元数据字典（补充source字段，统一占位符为N/A）
        metadata = {
            "create_time": (text_data.create_time or datetime.now()).isoformat(),
            "update_time": text_data.update_time.isoformat() if text_data.update_time else "N/A",
            "source": text_data.source or "N/A",  # 处理新增source字段，None转为N/A
            "states": text_data.states,
            "capabilities": text_data.capabilities,
            "device_id_clues": text_data.device_id_clues,
            "usage_habits": text_data.usage_habits,
            "others": text_data.others,
            **cleaned_other_meta
        }

        # 入库操作
        collection.add(
            ids=[text_data.text_id],
            documents=[text_data.content],
            metadatas=[metadata]
        )
        print(f"✅ 文本「{text_data.text_id}」已成功存入向量数据库")

    def search_topK_device_by_clues(self, clues: List[str], topk: int = 3) -> List[Dict[str, Any]]:
        """
        遍历向量库所有集合，采用调和平均聚合多线索相似度，返回综合匹配度最高的topk个集合
        新增：每个集合包含与各线索最相似的文档详情
        :param clues: 查询线索列表（如["床边的灯", "飞利浦"]）
        :param topk: 返回结果数量
        :return: 格式化的TopK集合结果（含各线索得分、综合调和得分、集合信息、各线索最优文档）
        """
        # 步骤1：输入校验
        if not clues or len(clues) == 0:
            raise ValueError("查询线索列表clues不能为空，请至少传入1个查询线索")
        n_clues = len(clues)  # 线索数量，用于调和平均计算

        # 步骤2：获取向量库中所有集合
        all_collections = self.client.list_collections()
        if not all_collections:
            print("⚠️  向量库中无任何集合，返回空结果")
            return []

        # 步骤3：遍历每个集合，计算其对所有线索的匹配距离+提取最优文档
        collection_score_map: Dict[str, Dict[str, Any]] = {}
        for collection in all_collections:
            coll_name = collection.name
            coll_metadata = collection.metadata or {}
            coll_doc_count = collection.count()  # 集合内文档数量

            # 3.1 初始化该集合的线索距离列表和最优文档列表
            coll_clue_distances = []
            coll_clue_best_docs = []  # 新增：存储各线索对应的最相似文档信息
            default_doc_info = {  # 空集合/无匹配时的默认文档信息
                "doc_id": "",
                "content": "",
                "metadata": {},
                "match_distance": self.default_distance
            }

            for clue in clues:
                # 3.2 空集合处理：无文档，直接添加默认值
                if coll_doc_count == 0:
                    coll_clue_distances.append(self.default_distance)
                    coll_clue_best_docs.append(default_doc_info)
                    continue

                boolean_where_filter = {"device_id_clues": {"$eq": True}}

                # 3.3 非空集合：查询该集合与当前线索的所有文档，获取完整信息（扩展include参数）
                query_results = collection.query(
                    query_texts=[clue],
                    where=boolean_where_filter,
                    n_results=coll_doc_count,  # 返回集合内所有文档
                    include=["documents", "metadatas", "distances"]  # 新增文档相关字段
                )

                # 3.4 提取查询结果中的文档信息和距离
                doc_ids = query_results["ids"][0]
                doc_contents = query_results["documents"][0]
                doc_metadatas = query_results["metadatas"][0]
                clue_distances = query_results["distances"][0]

                # 3.5 无匹配结果处理
                if not clue_distances or len(clue_distances) == 0:
                    coll_clue_distances.append(self.default_distance)
                    coll_clue_best_docs.append(default_doc_info)
                    continue

                # 3.6 找到最小距离对应的文档（核心：关联距离与文档）
                coll_min_distance = min(clue_distances)
                min_distance_idx = clue_distances.index(coll_min_distance)  # 找到最小距离的索引

                # 3.7 提取该索引对应的文档完整信息
                best_doc_info = {
                    "doc_id": doc_ids[min_distance_idx] if doc_ids else "",
                    "content": doc_contents[min_distance_idx] if doc_contents else "",
                    "metadata": doc_metadatas[min_distance_idx] if doc_metadatas else {},
                    "match_distance": coll_min_distance
                }

                # 3.8 保证距离>0，避免除零错误，添加到列表
                safe_distance = max(coll_min_distance, self.epsilon)
                coll_clue_distances.append(safe_distance)
                coll_clue_best_docs.append(best_doc_info)  # 新增：存入最优文档信息

            # 步骤4：计算该集合的调和平均综合得分
            reciprocal_sum = sum(1.0 / d for d in coll_clue_distances)
            synthetic_score = n_clues / reciprocal_sum if reciprocal_sum > 0 else float("inf")

            # 步骤5：存储该集合的完整信息（新增clue_best_docs字段）
            collection_score_map[coll_name] = {
                "collection_name": coll_name,  # 集合名（设备ID）
                "collection_metadata": coll_metadata,  # 集合元数据（含device_name）
                "document_count": coll_doc_count,  # 集合内文档数量
                "clue_distances": dict(zip(clues, coll_clue_distances)),  # 各线索的匹配距离（越小越相似）
                "clue_best_docs": dict(zip(clues, coll_clue_best_docs)),  # 新增：各线索对应的最相似文档详情
                "synthetic_score": synthetic_score  # 调和平均综合得分（越小越相似）
            }

        # 步骤6：按综合得分升序排序，取TopK
        sorted_collections = sorted(
            collection_score_map.values(),
            key=lambda x: x["synthetic_score"]
        )[:topk]

        # 步骤7：格式化返回结果
        return sorted_collections

    def get_device_multi_constraints_individual_match_scores(
            self,
            device_id: str,
            multi_clues: List[List[str]]
    ) -> Dict[Union[int, str], Dict[str, Any]]:
        """
        返回单个设备对多个约束条件各自对应的ChromaDB原始相似性距离（调和平均）+ 匹配文档内容
        说明：ChromaDB距离越小，代表设备与约束条件的匹配度越高；无匹配时返回默认距离1.0+空文档列表
        :param device_id: 智能家居设备的唯一标识ID（如Home Assistant设备ID）
        :param multi_clues: 设备匹配的约束条件/定位线索集合，外层列表包含多个独立约束条件，
                            每个约束条件对应一个内部字符串列表（存储该约束的具体线索内容）
        :return: Dict[Union[int, str], Dict[str, Any]] - 匹配度结果字典，
                 键：约束条件的索引（int）或线索组合摘要（str），
                 值：嵌套字典，包含两个字段：
                     1. harmonic_distance: 对应约束的调和平均原始相似性距离（无匹配返回1.0）
                     2. matching_documents: 对应约束的匹配文档列表（每个元素为文档详情字典，含doc_id/content/metadata/match_distance）
        """
        # 步骤1：严格输入校验
        if not device_id:
            print("⚠️  设备ID不能为空")
            return {}
        if not isinstance(multi_clues, list) or len(multi_clues) == 0:
            print("⚠️  约束条件集合multi_clues不能为空，且必须为嵌套列表")
            return {}

        # 步骤2：获取设备对应集合（处理集合不存在异常）
        try:
            collection = self.client.get_collection(
                name=device_id,
                embedding_function=self.embedding_func
            )
        except Exception as e:
            print(f"⚠️  设备ID「{device_id}」对应的集合不存在或获取失败：{e}")
            return {}

        # 步骤3：空集合处理（直接返回所有约束默认距离+空文档列表）
        coll_doc_count = collection.count()
        default_doc_list = []
        if coll_doc_count == 0:
            print(f"⚠️  设备ID「{device_id}」对应的集合无文档，返回默认距离+空文档")
            return {
                self._get_constraint_key(idx, constraint): {
                    "harmonic_distance": self.default_distance,
                    "matching_documents": default_doc_list
                }
                for idx, constraint in enumerate(multi_clues)
            }

        # 步骤4：初始化返回结果字典
        match_results = {}
        # 定义默认文档信息（无匹配时使用）
        default_doc_info = {
            "doc_id": "",
            "content": "",
            "metadata": {},
            "match_distance": self.default_distance
        }

        # 步骤5：遍历每个独立约束条件，调和平均计算距离+提取匹配文档
        for constraint_idx, constraint_clues in enumerate(multi_clues):
            # 5.1 单个约束内部线索校验（空线索直接返回默认值+空文档）
            if not isinstance(constraint_clues, list) or len(constraint_clues) == 0:
                constraint_key = self._get_constraint_key(constraint_idx, constraint_clues)
                match_results[constraint_key] = {
                    "harmonic_distance": self.default_distance,
                    "matching_documents": default_doc_list
                }
                continue

            # 5.2 构建查询过滤条件（复用现有device_id_clues标签，保持一致性）
            where_filter = {"device_id_clues": {"$eq": True}}

            # 5.3 初始化当前约束的距离列表和文档列表（去重存储）
            constraint_min_distances = []
            constraint_matching_docs = []
            doc_id_set = set()  # 用于文档去重，避免重复添加同一文档

            for clue in constraint_clues:
                if not clue:  # 空线索跳过
                    continue
                try:
                    # 5.4 查询该线索与集合内所有文档的匹配结果（含文档详情）
                    query_results = collection.query(
                        query_texts=[clue],
                        where=where_filter,
                        n_results=coll_doc_count,
                        include=["documents", "metadatas", "distances"]  # 提取文档内容和元数据
                    )

                    # 5.5 提取查询结果中的字段
                    doc_ids = query_results["ids"][0]
                    doc_contents = query_results["documents"][0]
                    doc_metadatas = query_results["metadatas"][0]
                    clue_distances = query_results["distances"][0]

                    # 5.6 无匹配结果处理
                    if not clue_distances or len(clue_distances) == 0:
                        continue

                    # 5.7 找到该线索的最小距离对应文档（最优匹配）
                    min_clue_distance = min(clue_distances)
                    min_distance_idx = clue_distances.index(min_clue_distance)
                    # 保证距离>0，避免后续调和平均除零
                    safe_min_distance = max(min_clue_distance, self.epsilon)

                    # 5.8 提取最优文档详情（避免索引越界）
                    doc_id = doc_ids[min_distance_idx] if (doc_ids and len(doc_ids) > min_distance_idx) else ""
                    doc_content = doc_contents[min_distance_idx] if (
                                doc_contents and len(doc_contents) > min_distance_idx) else ""
                    doc_metadata = doc_metadatas[min_distance_idx] if (
                                doc_metadatas and len(doc_metadatas) > min_distance_idx) else {}

                    # 5.9 文档去重：仅添加未出现过的文档
                    if doc_id not in doc_id_set and doc_id:
                        doc_id_set.add(doc_id)
                        single_doc_info = {
                            "doc_id": doc_id,
                            "content": doc_content,
                            "metadata": doc_metadata,
                            "match_distance": safe_min_distance,
                            "matching_clue": clue  # 标注该文档匹配的具体线索，便于追溯
                        }
                        constraint_matching_docs.append(single_doc_info)

                    # 5.10 存入该线索的最小安全距离
                    constraint_min_distances.append(safe_min_distance)

                except Exception as e:
                    print(f"⚠️  约束{constraint_idx}线索「{clue}」查询失败：{e}")
                    continue

            # 5.11 调和平均计算当前约束的最终原始距离
            if constraint_min_distances:
                n_valid_clues = len(constraint_min_distances)
                reciprocal_sum = sum(1.0 / d for d in constraint_min_distances)
                if reciprocal_sum > 0:
                    constraint_harmonic_distance = n_valid_clues / reciprocal_sum
                else:
                    constraint_harmonic_distance = self.default_distance
            else:
                constraint_harmonic_distance = self.default_distance
                # 无有效距离时，添加默认文档信息（便于调用者识别无匹配）
                constraint_matching_docs.append(default_doc_info)

            # 5.12 构建约束键，存入完整结果（距离+文档列表，保留4位小数）
            constraint_key = self._get_constraint_key(constraint_idx, constraint_clues)
            match_results[constraint_key] = {
                "harmonic_distance": round(constraint_harmonic_distance, 4),
                "matching_documents": constraint_matching_docs
            }

        # 步骤6：返回最终完整结果
        return match_results

    # 私有辅助函数：生成约束条件的唯一键（索引/摘要）
    def _get_constraint_key(self, idx: int, constraint_clues: List[str]) -> Union[int, str]:
        """
        生成约束条件的唯一键，优先返回线索拼接摘要，失败返回索引
        :param idx: 约束条件索引
        :param constraint_clues: 单个约束的线索列表
        :return: 约束键（str：线索摘要 / int：索引）
        """
        try:
            # 拼接线索为摘要（用「|」分隔，避免歧义）
            clue_summary = "|".join([str(clue).strip() for clue in constraint_clues if clue])
            return clue_summary if clue_summary else idx
        except Exception:
            return idx

    def print_all_collections_content(self):
        """
        格式化打印向量库中所有集合的基本信息，以及每个集合内的所有文档详情
        用于调试、数据验证和结果查看，格式清晰易读
        """
        print("=" * 80)
        print("📋 开始打印向量库所有集合及内容")
        print("=" * 80)

        # 步骤1：获取所有集合
        all_collections = self.client.list_collections()
        if not all_collections:
            print("⚠️  向量库中无任何集合，打印结束")
            print("=" * 80)
            return

        # 步骤2：遍历每个集合，打印详情
        for idx, collection in enumerate(all_collections, 1):
            coll_name = collection.name
            coll_metadata = collection.metadata or {}
            coll_doc_count = collection.count()

            # 打印集合基本信息
            print(f"\n【{idx}】集合基本信息")
            print(f"  - 集合名称（设备ID）：{coll_name}")
            print(f"  - 集合元数据：{coll_metadata}")
            print(f"  - 集合内文档数量：{coll_doc_count}")
            print(f"  - {'-' * 60}")

            # 步骤3：空集合处理，跳过文档打印
            if coll_doc_count == 0:
                print(f"  ⚠️  该集合无文档，跳过文档打印")
                continue

            # 步骤4：非空集合，获取所有文档（include指定返回所有字段）
            try:
                all_docs = collection.get(
                    include=["documents", "metadatas"]  # 包含文档ID、内容、元数据
                )
            except Exception as e:
                print(f"  ❌  获取该集合文档失败：{e}")
                continue

            # 步骤5：提取文档数据并格式化打印
            doc_ids = all_docs.get("ids", [])
            doc_contents = all_docs.get("documents", [])
            doc_metadatas = all_docs.get("metadatas", [])

            for doc_idx, (doc_id, doc_content, doc_meta) in enumerate(zip(doc_ids, doc_contents, doc_metadatas), 1):
                print(f"  【文档{doc_idx}】")
                print(f"    - 文档ID：{doc_id}")
                # 文档内容过长时截取前100字，避免打印冗余
                doc_content_show = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                print(f"    - 文档内容：{doc_content_show}")
                print(f"    - 文档元数据：{doc_meta or '无元数据'}")
                print(f"    {'-' * 50}")

        # 步骤6：打印结束标识
        print("\n" + "=" * 80)
        print("✅  向量库所有集合及内容打印完成")
        print("=" * 80)

    # ---------------------- 新增三个内容拼接函数 ----------------------
    def get_device_states_combined(self, device_id: str) -> str:
        """
        获取指定设备ID集合中，元数据states为True的所有文档内容，拼接为字符串返回
        :param device_id: 设备唯一标识ID（对应集合名称）
        :return: 拼接后的字符串（无匹配内容返回空字符串）
        """
        return self._get_device_field_combined(device_id, "states")

    def get_device_capabilities_combined(self, device_id: str) -> str:
        """
        获取指定设备ID集合中，元数据capabilities为True的所有文档内容，拼接为字符串返回
        :param device_id: 设备唯一标识ID（对应集合名称）
        :return: 拼接后的字符串（无匹配内容返回空字符串）
        """
        return self._get_device_field_combined(device_id, "capabilities")

    def get_device_usage_habits_combined(self, device_id: str) -> str:
        """
        获取指定设备ID集合中，元数据usage_habits为True的所有文档内容，拼接为字符串返回
        :param device_id: 设备唯一标识ID（对应集合名称）
        :return: 拼接后的字符串（无匹配内容返回空字符串）
        """
        return self._get_device_field_combined(device_id, "usage_habits")

    # 私有辅助函数：提取公共逻辑，避免代码冗余
    def _get_device_field_combined(self, device_id: str, field_name: str) -> str:
        """
        私有辅助函数：根据设备ID和字段名，筛选对应字段为True的内容并拼接
        :param device_id: 设备唯一标识ID
        :param field_name: 要筛选的元数字段名（states/capabilities/usage_habits）
        :return: 拼接后的字符串
        """
        # 步骤1：参数校验
        if not device_id or not field_name:
            return ""

        # 步骤2：获取设备对应集合（捕获集合不存在异常）
        try:
            collection = self.client.get_collection(
                name=device_id,
                embedding_function=self.embedding_func
            )
        except Exception:
            print(f"⚠️  设备ID「{device_id}」对应的集合不存在")
            return ""

        # 步骤3：空集合处理
        if collection.count() == 0:
            return ""

        # 步骤4：筛选对应字段为True的文档
        try:
            filtered_docs = collection.get(
                where={field_name: {"$eq": True}},  # 筛选条件：字段值为True
                include=["documents"]  # 仅获取文档内容，提升效率
            )
        except Exception as e:
            print(f"❌  筛选设备「{device_id}」字段「{field_name}」失败：{e}")
            return ""

        # 步骤5：提取内容并拼接（去重+过滤空内容）
        doc_contents = filtered_docs.get("documents", [])
        # 去重+过滤空字符串，避免冗余和无效内容
        unique_contents = list(filter(None, list(dict.fromkeys(doc_contents))))
        # 用「、」拼接，中文场景更易读
        return "、".join(unique_contents)

VECTORDB=VectorDB()

@tool
def search_topK_device_by_clues(clues: List[str]):
    """
    根据线索/约束条件找到最符合的设备
    :param clues:
    :param topk:
    :return:
    """
    pass

@tool
def add(device_id:str,content:str,tag:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag:
    :return:
    """
    pass

@tool
def update(device_id:str,old_content:str,new_content:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag:
    :return:
    """
    pass

@tool
def delete(device_id:str,content:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag:
    :return:
    """
    pass

@tool
def get_device_constraints_individual_match_scores(
    device_id: str,
    multi_clues: List[List[str]]
):
    """
    返回单个设备对多个约束条件各自对应的匹配度得分。

    :param device_id: 智能家居设备的唯一标识ID（如Home Assistant设备ID）。
    :param multi_clues: 设备匹配的约束条件/定位线索集合，外层列表包含多个独立约束条件，每个约束条件对应一个内部字符串列表（存储该约束的具体线索内容）。
    :return: Dict[Union[int, str], float] - 匹配度结果字典，键为约束条件的索引（或线索组合摘要），值为对应约束条件下设备的匹配度得分（通常取值范围0~1）。
    """
    # todo 核验VectorDB里面的实现，调用整理结果后返回
    pass

@tool
def get_device_all_states(device_id: str):
    """
    获取可以从设备ID查询到的所有状态信息
    :param device_id: 设备ID
    :return:
    """
    return VECTORDB.get_device_states_combined(device_id)

@tool
def get_device_all_capabilities(device_id: str):
    """
    获取可以从设备ID查询到的所有能力信息
    :param device_id: 设备ID
    :return:
    """
    return VECTORDB.get_device_capabilities_combined(device_id)

@tool
def get_device_all_usage_habits(device_id: str):
    """
    获取可以从设备ID查询到的所有使用习惯
    :param device_id: 设备ID
    :return:
    """
    return VECTORDB.get_device_usage_habits_combined(device_id)



def test_device_multi_constraints_match_pydantic():
    """
    测试函数（适配Pydantic版TextWithMeta）：验证get_device_multi_constraints_individual_match_scores的效果
    流程：初始化→创建设备集合→构造Pydantic文档→入库→多约束查询→解析结果
    """
    # ---------------------- 步骤1：初始化VectorDB实例 ----------------------
    vector_db = VectorDB()
    print("=" * 80)
    print("🚀 开始测试（适配Pydantic版TextWithMeta）设备多约束匹配功能")
    print("=" * 80)

    # ---------------------- 步骤2：定义测试设备信息 ----------------------
    test_device_id = "living_room_smart_device_001"  # 测试设备ID（对应Chroma集合名）
    test_device_name = "客厅智能设备组合"  # 测试设备名称
    # 创建/获取设备对应的Chroma集合
    test_collection = vector_db.get_or_create_collection(
        collection_name=test_device_id,
        device_name=test_device_name
    )

    # ---------------------- 步骤3：构造Pydantic版测试文档并入库（3条核心文档） ----------------------
    # 文档1：客厅智能吸顶灯（高匹配线索：客厅灯、暖光、智能吸顶灯）
    # 注意：text_id替代原doc_id，create_time使用Field默认值，device_id_clues=True确保过滤条件生效
    doc1 = TextWithMeta(
        text_id="text_001",  # 对应Pydantic的text_id字段
        content="客厅智能吸顶灯支持暖光/白光/中性光调节，亮度范围10-100%，可通过语音控制开启/关闭，当前处于暖光模式（亮度80%）。",
        device_id_clues=True,  # 关键：查询过滤条件依赖该字段为True
        capabilities=True,  # 标签字段赋值
        source="test_data",  # 可选字段赋值
        other_meta={"device_type": "ceiling_light", "location": "living_room"}  # 自定义元信息
    )

    # 文档2：客厅自动加湿器（中等匹配线索：加湿器、自动开关、客厅）
    doc2 = TextWithMeta(
        text_id="text_002",
        content="客厅落地式加湿器支持自动开关功能，当环境湿度低于40%时自动开启，高于60%时自动关闭，水箱容量5L，当前湿度45%。",
        device_id_clues=True,
        capabilities=True,
        source="test_data",
        other_meta={"device_type": "humidifier", "location": "living_room"}
    )

    # 文档3：卧室变频空调（低匹配线索：卧室、空调、变频）
    doc3 = TextWithMeta(
        text_id="text_003",
        content="卧室变频空调支持冷暖切换，能效等级1级，设定温度25℃，当前处于制冷静音模式，风速自动调节。",
        device_id_clues=True,
        states=True,  # 标签字段赋值
        source="test_data",
        other_meta={"device_type": "air_conditioner", "location": "bedroom"}
    )

    # 批量入库测试文档（适配Pydantic实例）
    test_docs = [doc1, doc2, doc3]
    for doc in test_docs:
        vector_db.add_text_to_vector_db(text_data=doc, collection=test_collection)
    print("=" * 80)

    # ---------------------- 步骤4：定义测试多约束条件（3个约束，覆盖高/中/低匹配） ----------------------
    test_multi_clues = [
        # 约束0：客厅智能暖光吸顶灯（高匹配，对应text_001）
        ["客厅灯", "暖光", "智能吸顶灯"],
        # 约束1：卧室变频空调（低匹配，对应text_003，设备ID是客厅设备，匹配度低）
        ["卧室", "空调", "变频"],
        # 约束2：客厅自动开关加湿器（中等匹配，对应text_002）
        ["加湿器", "自动开关", "客厅"]
    ]
    print(f"📋 定义的测试多约束条件：")
    for idx, constraint in enumerate(test_multi_clues):
        print(f"  约束{idx}：{constraint}")
    print("=" * 80)

    # ---------------------- 步骤5：调用目标函数进行多约束匹配查询 ----------------------
    print("🔍 开始执行多约束匹配查询...")
    match_results = vector_db.get_device_multi_constraints_individual_match_scores(
        device_id=test_device_id,
        multi_clues=test_multi_clues
    )

    # ---------------------- 步骤6：格式化解析并打印结果 ----------------------
    print("✅ 多约束匹配查询完成，开始解析结果")
    print("=" * 80)
    if not match_results:
        print("❌ 未获取到匹配结果")
        return

    for constraint_key, result_detail in match_results.items():
        # 提取核心结果字段
        harmonic_distance = result_detail["harmonic_distance"]
        matching_docs = result_detail["matching_documents"]
        doc_count = len(matching_docs)

        # 打印约束整体信息
        print(f"\n📌 约束结果：{constraint_key}")
        print(f"  ├─ 调和平均原始距离：{harmonic_distance}（越小匹配度越高）")
        print(f"  └─ 匹配文档数量：{doc_count}")

        # 打印每个匹配文档的详情
        if doc_count > 0:
            for doc_idx, doc_info in enumerate(matching_docs, 1):
                print(f"\n  📄 文档{doc_idx}详情：")
                print(f"    ├─ 文档ID（text_id）：{doc_info['doc_id']}")
                print(f"    ├─ 匹配线索：{doc_info['matching_clue']}")
                print(f"    ├─ 线索匹配距离：{doc_info['match_distance']:.4f}")
                # 文档内容过长时截取前100字
                content_show = doc_info['content'][:100] + "..." if len(doc_info['content']) > 100 else doc_info['content']
                print(f"    ├─ 文档内容：{content_show}")
                print(f"    └─ 文档元数据：{doc_info['metadata']}")
        print("-" * 60)

    print("=" * 80)
    print("🎉 （适配Pydantic版）设备多约束匹配功能测试完成")
    print("=" * 80)

# ---------------------- 执行测试 ----------------------
if __name__ == "__main__":
    test_device_multi_constraints_match_pydantic()




def old_test_01():
    # 步骤1：初始化VectorDB实例
    vector_db = VectorDB()
    print("=== 初始化VectorDB完成，开始构造测试数据 ===")

    # 步骤2：定义当前时间（用于赋值create_time/update_time）
    current_time = datetime.now()
    update_time = datetime.now()

    # 步骤3：构造2个设备集合，每个集合存入2条测试数据
    ## 3.1 设备1：集合名（设备ID）= "DEVICE_001"，设备名称= "飞利浦床边位置传感器"
    coll_001 = vector_db.get_or_create_collection(
        collection_name="DEVICE_001",
        device_name="飞利浦床边位置传感器"
    )
    # 构造DEVICE_001的测试数据1
    text_001_01 = TextWithMeta(
        text_id="DEVICE_001_doc_01",
        content="飞利浦床边位置传感器：支持人体红外感应，靠近自动唤醒，在线运行稳定",
        states=True,  # 在线状态
        capabilities=True,  # 具备位置感知能力
        device_id_clues=True,  # 包含床边设备标识
        usage_habits=True,  # 符合床边使用习惯
        others=True,  # 适配飞利浦生态
        create_time=current_time,
        update_time=update_time,
        source="飞利浦官网",
        other_meta={"model": "PH-Bed001", "price": 199.99}
    )
    # 构造DEVICE_001的测试数据2
    text_001_02 = TextWithMeta(
        text_id="DEVICE_001_doc_02",
        content="飞利浦床边传感器维护说明：定期清洁感应窗口，避免遮挡影响精度",
        states=False,  # 离线（维护状态）
        capabilities=True,
        device_id_clues=True,
        usage_habits=True,
        others=True,
        create_time=current_time,
        source="飞利浦售后手册",
        other_meta={"maintain_cycle": "3个月", "contact": "400-888-8888"}
    )
    # 存入DEVICE_001集合
    vector_db.add_text_to_vector_db(text_001_01, coll_001)
    vector_db.add_text_to_vector_db(text_001_02, coll_001)

    ## 3.2 设备2：集合名（设备ID）= "DEVICE_002"，设备名称= "小米客厅普通吸顶灯"
    coll_002 = vector_db.get_or_create_collection(
        collection_name="DEVICE_002",
        device_name="小米客厅普通吸顶灯"
    )
    # 构造DEVICE_002的测试数据1（无床边标识，过滤时会被排除）
    text_002_01 = TextWithMeta(
        text_id="DEVICE_002_doc_01",
        content="小米客厅吸顶灯：遥控调光，色温可调，离线待机功耗低",
        states=False,
        capabilities=False,  # 无位置感知能力
        device_id_clues=False,  # 无床边设备标识
        usage_habits=False,  # 不符合床边使用习惯
        others=False,
        create_time=current_time,
        source="小米商城",
        other_meta={"model": "MI-Light005", "max_brightness": "500流明"}
    )
    # 存入DEVICE_002集合
    vector_db.add_text_to_vector_db(text_002_01, coll_002)

    # 步骤4：构造查询线索，执行TopK检索（topk=2）
    query_clues = ["床边位置感知设备", "飞利浦在线设备"]
    print("\n=== 开始执行多线索检索 ===")
    print(f"查询线索：{query_clues}")
    print(f"返回TopK数量：2")
    try:
        topk_results = vector_db.search_topK_device_by_clues(
            clues=query_clues,
            topk=2
        )
    except Exception as e:
        print(f"❌ 检索失败：{e}")
        topk_results = []

    # 步骤5：格式化打印检索结果
    print("\n=== 检索结果（TopK）解析 ===")
    if not topk_results:
        print("⚠️  无符合条件的检索结果")
    else:
        for idx, result in enumerate(topk_results, 1):
            print(f"\n【第 {idx} 条结果（综合相似度第 {idx}）】")
            print(f"  1. 集合信息（设备ID）：{result['collection_name']}")
            print(f"  2. 设备名称：{result['collection_metadata'].get('device_name', 'N/A')}")
            print(f"  3. 集合内文档数量：{result['document_count']}")
            print(f"  4. 综合调和得分（越小越相似）：{result['synthetic_score']:.6f}")

            print(f"  5. 各线索匹配距离（越小越相似）：")
            for clue, distance in result['clue_distances'].items():
                print(f"     - 线索「{clue}」：{distance:.6f}")

            print(f"  6. 各线索最优匹配文档详情：")
            for clue, doc_info in result['clue_best_docs'].items():
                print(f"     - 线索「{clue}」最优文档：")
                print(f"       > 文档ID：{doc_info['doc_id']}")
                print(f"       > 文档内容：{doc_info['content'][:50]}..." if len(
                    doc_info['content']) > 50 else f"       > 文档内容：{doc_info['content']}")
                print(f"       > 文档匹配距离：{doc_info['match_distance']:.6f}")
                print(f"       > 文档元数据（布尔标签）：")
                doc_meta = doc_info['metadata']
                print(f"         - 在线状态（states）：{doc_meta.get('states', 'N/A')}")
                print(f"         - 位置感知能力（capabilities）：{doc_meta.get('capabilities', 'N/A')}")
                print(f"         - 床边标识（device_id_clues）：{doc_meta.get('device_id_clues', 'N/A')}")
