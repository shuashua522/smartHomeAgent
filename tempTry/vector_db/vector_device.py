from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection



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

    create_time: datetime  # 创建时间（元信息）
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

VECTORDB=VectorDB()

# 3. 测试流程：入库 + 检索验证
if __name__ == "__main__":
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
