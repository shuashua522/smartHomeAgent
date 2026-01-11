import os
import uuid

from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection
from langchain.tools import tool

from smartHome.m_agent.common.get_llm import get_llm

def get_short_uuid_by_cut(length: int = 12) -> str:
    """
    截取uuid的hex值生成短字符串
    :param length: 要生成的短字符串长度（建议4-16位，过长失去缩短意义，过短易冲突）
    :return: 简短字符串（十六进制，0-9a-f）
    """
    if not (4 <= length <= 32):
        raise ValueError("长度建议在4-32之间")
    full_hex = uuid.uuid4().hex
    return full_hex[:length]  # 截取前length位（也可截取后length位：full_hex[-length:]）
# 1. 数据模型（保持不变）
class TextWithMeta(BaseModel):
    """单条文本的数据模型（含内容、多标签、元信息）"""
    text_id: str = Field(default_factory=lambda: get_short_uuid_by_cut(12),description="文本唯一标识（方便后续更新/删除）")
    content: str  # 核心文本内容（用于生成向量）

    # 标签，表示这个content是什么类型的信息
    states: bool=False
    capabilities: bool=False
    device_id_clues: bool=False
    usage_habits: bool=False
    others: bool=False

    # 修正1：create_time 用Field设置实时默认值（lambda确保实例化时实时计算）
    create_time: datetime = Field(default_factory=lambda: datetime.now(), description="创建时间")
    update_time: datetime = Field(default_factory=lambda: datetime.now(), description="更新时间")  # 更新时间（元信息，可选）
    source:  Optional[str] = None
    other_meta: Optional[dict] = None  # 其他自定义元信息（如作者、来源等）

class VectorDB():
    def __init__(self):
        # 初始化文本嵌入函数（保持不变）
        self.embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 轻量高效，支持中英文
        )
        # 初始化Chroma向量数据库（保持不变，支持持久化）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "chroma_text_db")
        self.client = chromadb.PersistentClient(path=file_path)
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

    def retrieve_similar_content(self,collection_name: str,old_content: str,topk: int = 5,tag:str="device_id_clues") -> List[Dict]:
        """
        从指定集合中检索与old_content最相似的TopK条内容
        :param collection_name: 目标集合名称（设备ID）
        :param old_content: 待匹配的原始文本内容
        :param topk: 要返回的最相似结果条数，默认5
        :return: 格式化的相似结果列表，每个元素包含id、content、distance、metadata（距离越小越相似）
        """
        # 步骤1：输入参数校验
        if not isinstance(collection_name, str) or len(collection_name.strip()) == 0:
            raise ValueError("集合名称（collection_name）不能为空且必须为字符串")
        if not isinstance(old_content, str) or len(old_content.strip()) == 0:
            raise ValueError("待匹配文本（old_content）不能为空且必须为字符串")
        if not isinstance(topk, int) or topk <= 0:
            raise ValueError("TopK（topk）必须为正整数")

        # 步骤2：获取目标集合（复用已有方法，若集合不存在则创建空集合）
        target_collection = self.get_or_create_collection(collection_name=collection_name)

        # 步骤3：判断集合是否有文档，无文档直接返回空列表
        if target_collection.count() == 0:
            print(f"提示：集合「{collection_name}」中无任何文档，无法进行相似性检索")
            return []

        # 步骤4：执行相似性检索（Chroma的query方法）
        try:
            # query_texts：传入待匹配文本列表（单文本检索传列表包裹）
            # n_results：返回最相似的topk条结果
            # include：指定返回的字段（ids=文档ID、documents=文档内容、distances=匹配距离、metadatas=文档元数据）
            boolean_where_filter = {"device_id_clues": {"$eq": True}}
            retrieve_result = target_collection.query(
                query_texts=[old_content.strip()],
                where=boolean_where_filter,
                n_results=topk,
                include=["documents", "distances", "metadatas"]
            )
        except Exception as e:
            raise RuntimeError(f"相似性检索失败：{str(e)}") from e

        # 步骤5：格式化检索结果（Chroma返回结果为字典，需整理为易使用的列表）
        formatted_results = []
        # Chroma返回的每个字段都是二维列表（对应多个query_texts），这里取索引0（单文本检索）
        ids = retrieve_result.get("ids", [[]])[0]
        documents = retrieve_result.get("documents", [[]])[0]
        distances = retrieve_result.get("distances", [[]])[0]
        metadatas = retrieve_result.get("metadatas", [[]])[0]

        # 遍历结果，拼接为结构化字典
        for doc_id, content, distance, meta in zip(ids, documents, distances, metadatas):
            # 处理距离异常值（保证大于epsilon，与类中定义一致）
            safe_distance = max(distance, self.epsilon) if distance is not None else self.default_distance
            # 处理元数据为None的情况
            safe_meta = meta if isinstance(meta, dict) else {}

            formatted_results.append({
                "doc_id": doc_id,  # 文档唯一ID
                "content": content,  # 文档核心内容
                # "distance": safe_distance,  # 匹配距离（越小越相似）
                # "metadata": safe_meta  # 文档元数据（如创建时间、标签等）
            })

        # 步骤6：返回格式化结果（无匹配结果时返回空列表）
        return formatted_results

    def search_topK_device_by_clues(self, clues: List[str], topk: int = 3) -> List[Dict[str, Any]]:
        """
        遍历向量库所有设备，采用调和平均聚合多线索相似度，返回综合匹配度最高的topk个设备
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

    def update_document_content(self,
                                collection_name: str,
                                doc_id: str,
                                new_content: str) -> str:
        """
        更新指定集合中指定doc_id的文档内容为new_content（自动重新生成向量）
        :param collection_name: 目标集合名称（设备ID）
        :param doc_id: 待更新文档的唯一ID
        :param new_content: 文档的新内容（替换原有内容）
        :return: 更新结果说明
        """
        # 步骤1：输入参数合法性校验
        if not isinstance(collection_name, str) or len(collection_name.strip()) == 0:
            raise ValueError("集合名称（collection_name）不能为空且必须为字符串")
        if not isinstance(doc_id, str) or len(doc_id.strip()) == 0:
            raise ValueError("文档ID（doc_id）不能为空且必须为字符串")
        if not isinstance(new_content, str) or len(new_content.strip()) == 0:
            raise ValueError("新文档内容（new_content）不能为空且必须为字符串")

        # 步骤2：获取目标集合（复用已有方法）
        target_collection = self.get_or_create_collection(collection_name=collection_name)

        # 步骤3：校验集合是否有文档，无文档直接返回失败结果
        if target_collection.count() == 0:
            return f"更新失败：集合「{collection_name}」中无任何文档，不存在文档「{doc_id}」"
        # 步骤4：校验待更新文档（doc_id）是否存在【核心修复部分】
        try:
            # 修复点1：去掉include=["ids"]，改为合法的include（或不指定include，默认返回ids+其他必要字段）
            # 仅传入ids=[doc_id.strip()]，查询指定文档，include传入["documents"]（合法字段，仅为验证存在性）
            existing_doc = target_collection.get(
                ids=[doc_id.strip()],
                include=["documents"]  # 传入合法字段，无需指定ids（返回结果默认包含ids）
            )
        except Exception as e:
            raise RuntimeError(f"查询待更新文档失败：{str(e)}") from e

        # 修复点2：判断返回的ids列表是否非空，验证文档是否存在
        existing_ids = existing_doc.get("ids", [])
        if not existing_ids or len(existing_ids) == 0:
            return f"更新失败：集合「{collection_name}」中不存在文档「{doc_id}」"

        # 步骤5：执行文档内容更新（Chroma自动重新生成向量）
        try:
            target_collection.update(
                ids=[doc_id.strip()],  # 指定待更新的文档ID（列表格式，支持批量更新）
                documents=[new_content.strip()]  # 新文档内容（与ids一一对应）
            )
        except Exception as e:
            raise RuntimeError(f"文档内容更新失败：{str(e)}") from e

        # 步骤6：返回成功结果
        return f"更新成功：集合「{collection_name}」中的文档「{doc_id}」内容已替换为新内容"

    def delete_document(self,collection_name: str,doc_id: str) -> str:
        """
        从指定集合中精准删除doc_id对应的文档（完整移除内容、向量、元数据）
        :param collection_name: 目标集合名称（设备ID）
        :param doc_id: 待删除文档的唯一ID
        :return: 格式化的删除结果字典，包含是否成功、提示信息、删除详情
        """
        # 步骤1：输入参数合法性校验
        if not isinstance(collection_name, str) or len(collection_name.strip()) == 0:
            raise ValueError("集合名称（collection_name）不能为空且必须为字符串")
        if not isinstance(doc_id, str) or len(doc_id.strip()) == 0:
            raise ValueError("文档ID（doc_id）不能为空且必须为字符串")

        # 步骤2：获取目标集合（复用已有方法，集合不存在则创建空集合，后续校验会拦截）
        target_collection = self.get_or_create_collection(collection_name=collection_name)

        # 步骤3：校验集合是否有文档，无文档直接返回失败结果
        if target_collection.count() == 0:
            return f"删除失败：集合「{collection_name}」中无任何文档，不存在文档「{doc_id}」"

        # 步骤4：校验待删除文档（doc_id）是否存在（沿用修复后的get方法逻辑，避免报错）
        try:
            # 省略include参数，默认返回ids；或指定合法include字段，仅为验证存在性
            existing_doc = target_collection.get(ids=[doc_id.strip()])
        except Exception as e:
            raise RuntimeError(f"查询待删除文档失败：{str(e)}") from e

        existing_ids = existing_doc.get("ids", [])
        if not existing_ids or len(existing_ids) == 0:
            return f"删除失败：集合「{collection_name}」中不存在文档「{doc_id}」"

        # 步骤5：执行文档删除（Chroma的delete方法，通过ids精准删除，支持批量）
        try:
            target_collection.delete(
                ids=[doc_id.strip()]  # 列表格式，支持批量删除（如["doc_001", "doc_002"]）
            )
        except Exception as e:
            raise RuntimeError(f"文档删除失败：{str(e)}") from e

        # 步骤6：返回格式化的成功结果
        return f"删除成功：集合「{collection_name}」中的文档「{doc_id}」已被完整移除"

    def get_device_multi_constraints_individual_match_scores(
            self,
            device_id: str,
            multi_clues: List[List[str]],
            topk: int = 3  # 新增：每个线索匹配的前topk个内容
    ) -> Dict[Union[int, str], Dict[str, Any]]:
        """
        返回单个设备对多个约束条件的匹配结果：
        1. 每个约束组内的单个线索各自匹配前topk个内容
        2. 对约束组内所有线索的匹配结果按文档ID去重
        说明：ChromaDB距离越小，代表设备与约束条件的匹配度越高；无匹配时返回默认空文档
        :param device_id: 智能家居设备的唯一标识ID
        :param multi_clues: 设备匹配的约束条件集合，外层列表=多个约束组，内层列表=单个约束组的具体线索
        :param topk: 每个线索匹配返回的前topk个内容（默认5）
        :return: 匹配结果字典，结构：
                 键：约束组的索引/摘要字符串
                 值：嵌套字典，包含两个字段：
                     1. individual_clue_matches: 字典（键=线索，值=该线索匹配的前topk个文档列表）
                     2. unique_matching_documents: 该约束组所有线索匹配结果去重后的文档列表（含匹配的所有线索）
        """
        # 步骤1：严格输入校验（新增topk校验）
        if not device_id:
            print("⚠️  设备ID不能为空")
            return {}
        if not isinstance(multi_clues, list) or len(multi_clues) == 0:
            print("⚠️  约束条件集合multi_clues不能为空，且必须为嵌套列表")
            return {}
        if not isinstance(topk, int) or topk <= 0:
            print("⚠️  topk必须为正整数")
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

        # 步骤3：空集合处理（直接返回所有约束默认空结果）
        coll_doc_count = collection.count()
        if coll_doc_count == 0:
            print(f"⚠️  设备ID「{device_id}」对应的集合无文档，返回空匹配结果")
            default_unique_docs = [{
                "doc_id": "",
                "content": "",
                "metadata": {},
                "match_distance": self.default_distance,
                "matching_clues": []
            }]
            return {
                self._get_constraint_key(idx, constraint): {
                    "individual_clue_matches": {},
                    "unique_matching_documents": default_unique_docs
                }
                for idx, constraint in enumerate(multi_clues)
            }

        # 步骤4：初始化返回结果字典
        match_results = {}

        # 步骤5：遍历每个约束组，计算单个线索topk匹配 + 组内去重
        for constraint_idx, constraint_clues in enumerate(multi_clues):
            # 5.1 空线索组处理
            if not isinstance(constraint_clues, list) or len(constraint_clues) == 0:
                constraint_key = self._get_constraint_key(constraint_idx, constraint_clues)
                match_results[constraint_key] = {
                    "individual_clue_matches": {},
                    "unique_matching_documents": [{
                        "doc_id": "",
                        "content": "",
                        "metadata": {},
                        "match_distance": self.default_distance,
                        "matching_clues": []
                    }]
                }
                continue

            # 5.2 构建查询过滤条件（保持原有逻辑）
            where_filter = {"device_id_clues": {"$eq": True}}

            # 5.3 初始化存储：
            # - individual_clue_matches: 每个线索的topk匹配结果
            # - doc_id_to_info: 按doc_id去重，记录文档匹配的所有线索
            individual_clue_matches = {}
            doc_id_to_info = {}

            for clue in constraint_clues:
                if not clue:  # 空线索跳过
                    individual_clue_matches[clue] = []
                    continue

                try:
                    # 5.4 查询该线索的前topk个匹配结果（核心修改：n_results=topk）
                    query_results = collection.query(
                        query_texts=[clue],
                        where=where_filter,
                        n_results=topk,  # 只取前topk个，替代原有的全量查询
                        include=["documents", "metadatas", "distances"]
                    )

                    # 5.5 提取查询结果字段
                    doc_ids = query_results["ids"][0]
                    doc_contents = query_results["documents"][0]
                    doc_metadatas = query_results["metadatas"][0]
                    clue_distances = query_results["distances"][0]

                    # 5.6 整理该线索的topk匹配结果
                    clue_topk_matches = []
                    for i in range(len(doc_ids)):
                        # 防止索引越界
                        doc_id = doc_ids[i] if i < len(doc_ids) else ""
                        doc_content = doc_contents[i] if i < len(doc_contents) else ""
                        doc_metadata = doc_metadatas[i] if i < len(doc_metadatas) else {}
                        match_distance = clue_distances[i] if i < len(clue_distances) else self.default_distance
                        safe_distance = max(match_distance, self.epsilon)

                        # 单个文档信息
                        doc_info = {
                            "doc_id": doc_id,
                            "content": doc_content,
                            "metadata": doc_metadata,
                            "match_distance": safe_distance
                        }
                        clue_topk_matches.append(doc_info)

                        # 5.7 去重逻辑：按doc_id合并，记录匹配的所有线索
                        if doc_id and doc_id != "":
                            if doc_id not in doc_id_to_info:
                                doc_id_to_info[doc_id] = {
                                    "doc_id": doc_id,
                                    "content": doc_content,
                                    "metadata": doc_metadata,
                                    "match_distance": safe_distance,
                                    "matching_clues": [clue]  # 记录匹配到的线索
                                }
                            else:
                                # 追加匹配线索（避免重复）
                                if clue not in doc_id_to_info[doc_id]["matching_clues"]:
                                    doc_id_to_info[doc_id]["matching_clues"].append(clue)

                    # 5.8 保存该线索的topk结果
                    individual_clue_matches[clue] = clue_topk_matches

                except Exception as e:
                    print(f"⚠️  约束{constraint_idx}线索「{clue}」查询失败：{e}")
                    individual_clue_matches[clue] = []
                    continue

            # 5.9 处理去重后的文档列表（无匹配则返回默认值）
            unique_matching_docs = list(doc_id_to_info.values()) if doc_id_to_info else [{
                "doc_id": "",
                "content": "",
                "metadata": {},
                "match_distance": self.default_distance,
                "matching_clues": []
            }]

            # 5.10 构建约束组结果（移除调和平均距离，新增单个线索/去重结果）
            constraint_key = self._get_constraint_key(constraint_idx, constraint_clues)
            match_results[constraint_key] = {
                "individual_clue_matches": individual_clue_matches,
                "unique_matching_documents": unique_matching_docs
            }

        # 步骤6：返回最终结果
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

        return f"{device_id}({collection.metadata['device_name']}):{'、'.join(unique_contents)}"

VECTORDB=VectorDB()


def format_collections_to_string(sorted_collections):
    """
    格式化 VECTORDB.search_topK_device_by_clues()的结果成字符串，以提供给LLM
    :param sorted_collections:
    :return:
    """
    # 初始化最终的格式化字符串（用列表存储各设备信息，最后拼接，效率高于字符串直接累加）
    result_lines = []
    # 添加上下文标题，提升可读性
    result_lines.append("=== TopK 最优设备匹配结果汇总 ===")

    # 步骤1：遍历sorted_collections中的每个设备（嵌套字典）
    for idx, collection in enumerate(sorted_collections, start=1):
        device_id = collection["collection_name"]

        clue_best_docs = collection["clue_best_docs"]
        # 3.2 格式化线索与最佳匹配文本（遍历clue_best_docs，展示对应关系）
        result_lines.append(f"{device_id}与各线索的最佳匹配文本详情：")
        for clue, best_doc in clue_best_docs.items():
            # 处理最佳文本可能为None/空字符串的边界情况，避免展示混乱
            best_content=best_doc["content"]
            doc_content = best_content if best_content else "无匹配有效文本"
            # 缩进展示，提升可读性
            result_lines.append(f"  - 线索「{clue}」：{doc_content}")

    # 步骤4：将列表中的所有行拼接为一个完整字符串（用换行符\n连接）
    final_formatted_str = "\n".join(result_lines)

    return final_formatted_str
@tool
def search_topK_device_by_clues(clues: List[str]):
    """
    根据线索/约束条件找到最符合的设备
    :param clues:
    :param topk:
    :return:
    """
    sorted_collections=VECTORDB.search_topK_device_by_clues(clues=clues,topk=3)
    topk_devices_str=format_collections_to_string(sorted_collections)
    prompt = """
            找到最符合线索/约束条件的设备，仅返回最佳的设备ID
            """

    agent = create_agent(model=get_llm())
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
    return result["messages"][-1].content

@tool
def add(device_id:str,content:str,tag:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag: 取值为【capabilities，device_id_clues，usage_habits】中的任一个。
    capabilities说明content是设备能力的补充；device_id_clues说明content是从多个设备中找到该设备的线索；usage_habits说明content是设备的使用习惯
    :return:
    """
    text_instance=TextWithMeta(
        content=content
    )
    setattr(text_instance, tag, True)
    VECTORDB.add_text_to_vector_db(text_instance,VECTORDB.get_or_create_collection(device_id))
    return "添加成功"

@tool
def tool_update_doc_content(device_id: str,doc_id: str,new_content: str):
    """
    更新指定设备集合中指定doc_id的文档内容为new_content
    """
    return VECTORDB.update_document_content(device_id,doc_id,new_content)
@tool
def update(device_id:str,old_content:str,new_content:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag:
    :return:
    """
    retrieve_result=VECTORDB.retrieve_similar_content(collection_name=device_id, old_content=old_content)
    prompt = f"""
    根据检索结果找到与{old_content}最相似的doc_id，然后调用工具将内容更新为{new_content}
    【检索结果】:{retrieve_result}
    """

    agent = create_agent(model=get_llm(),tools=[tool_update_doc_content],)
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
    return result["messages"][-1].content

@tool
def tool_delete_doc_content(device_id: str,doc_id: str):
    """
    从指定集合中精准删除doc_id对应的文档
    """
    return VECTORDB.delete_document(collection_name=device_id,doc_id=doc_id)
@tool
def delete(device_id:str,content:str):
    """
    添加事实信息
    :param device_id:
    :param content:
    :param tag:
    :return:
    """
    retrieve_result = VECTORDB.retrieve_similar_content(collection_name=device_id, old_content=content)
    prompt = f"""
        根据检索结果找到与{content}最相似的doc_id，然后调用工具将其删除
        【检索结果】:{retrieve_result}
        """

    agent = create_agent(model=get_llm(), tools=[tool_delete_doc_content], )
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
    return result["messages"][-1].content

@tool
def get_device_constraints_individual_match_text(
    device_id: str,
    multi_clues: List[List[str]]
)->str:
    """
    返回记忆库中该设备对多个约束条件各自对应的匹配文本。
    :param device_id: 智能家居设备的唯一标识ID（如Home Assistant设备ID）。
    :param multi_clues: 设备匹配的约束条件/定位线索集合，外层列表包含多个独立约束条件，每个约束条件对应一个内部字符串列表（存储该约束的具体线索内容）。
    """
    search_result=VECTORDB.get_device_multi_constraints_individual_match_scores(device_id=device_id,multi_clues=multi_clues)
    ans_str_list = []
    for key, val in search_result.items():
        content_str = ",".join([item["content"] for item in val["unique_matching_documents"]])
        clue_result_str = f"对于线索/约束：{key},匹配到的内容为:{content_str})"
        ans_str_list.append(clue_result_str)

    return "\n".join(ans_str_list)

@tool
def get_device_all_states()->str:
    """
    获取可以从家中所有设备，其各自能查询到的所有状态信息
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list=[]
    for collection in all_collections:
        device_id = collection.name
        ans_str_list.append(VECTORDB.get_device_states_combined(device_id=device_id))
    return "\n".join(ans_str_list)

@tool
def get_devices_states(device_ids:list[str])->str:
    """
        获取可以从给定设备列表，其各自能查询到的所有状态类型
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        if device_id not in device_ids:
            continue
        ans_str_list.append(VECTORDB.get_device_states_combined(device_id=device_id))
    return "\n".join(ans_str_list)

@tool
def get_device_all_capabilities()->str:
    """
    获取可以从家中所有设备，其各自能查询到的所有能力信息
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        ans_str_list.append(VECTORDB.get_device_capabilities_combined(device_id=device_id))
    return "\n".join(ans_str_list)
@tool
def get_devices_capabilities(device_ids:list[str])->str:
    """
    获取可以从给定设备列表，其各自能查询到的所有能力信息
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        if device_id not in device_ids:
            continue
        ans_str_list.append(VECTORDB.get_device_capabilities_combined(device_id=device_id))
    return "\n".join(ans_str_list)
@tool
def get_device_all_usage_habits()->str:
    """
    获取可以从家中所有设备，其各自能查询到的所有使用习惯
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        ans_str_list.append(VECTORDB.get_device_usage_habits_combined(device_id=device_id))
    return "\n".join(ans_str_list)

@tool
def get_devices_usage_habits(device_ids:list[str])->str:
    """
    取可以从给定设备列表，其各自能查询到的所有使用习惯
    """
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        if device_id not in device_ids:
            continue
        ans_str_list.append(VECTORDB.get_device_usage_habits_combined(device_id=device_id))
    return "\n".join(ans_str_list)

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
    # test_device_multi_constraints_match_pydantic()
    # VECTORDB.print_all_collections_content()
    device_ids=["164c1a92b8ce9cda0e2a8c13440b4722"]
    all_collections = VECTORDB.client.list_collections()
    ans_str_list = []
    for collection in all_collections:
        device_id = collection.name
        if device_id not in device_ids:
            continue
        ans_str_list.append(VECTORDB.get_device_states_combined(device_id=device_id))
    print("\n".join(ans_str_list))
    # result=VECTORDB.search_topK_device_by_clues(clues=["灯泡","调色温"],topk=3)
    # print(format_collections_to_string(result))
    """
    ⚠️  设备ID「28adb3b1-b520-4c5b-8b13-8b93bdfa5d5c」对应的集合不存在
⚠️  设备ID「7ff9f9cc-c531-4d3f-939e-b95386d6f7b2」对应的集合不存在
⚠️  设备ID「9bde1df2-dfcb-4966-a6a3-3026fa17fd77」对应的集合不存在
    """

    # text=TextWithMeta(
    #     content="df406d66e297203b9cbccd7f7b2b0376",
    #     device_id_clues=True
    # )
    # VECTORDB.add_text_to_vector_db(text_data=text,collection=VECTORDB.get_or_create_collection("df406d66e297203b9cbccd7f7b2b0376"))
    # all_collections = VECTORDB.client.list_collections()
    # ans_str_list = []
    # for collection in all_collections:
    #     device_id = collection.name
    #     ans_str_list.append(VECTORDB.get_device_states_combined(device_id=device_id))
    # print("\n".join(ans_str_list))



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
