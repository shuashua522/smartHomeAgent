import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def chromadb_boolean_metadata_demo():
    # 步骤 1：初始化环境（适配 ChromaDB 1.4.0 稳定版）
    print("=== 初始化 ChromaDB 环境 ===")
    # 1.1 确认版本
    print(f"ChromaDB 版本：{chromadb.__version__}")
    # 1.2 初始化持久化客户端
    client = chromadb.PersistentClient(path="./chroma_boolean_metadata")
    # 1.3 定义默认嵌入函数
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # 轻量高效，支持中英文
    )

    # 步骤 2：创建集合（无特殊配置，支持布尔元数据）
    collection_name = "smart_device_boolean_metadata"
    try:
        # 若集合已存在，先删除（方便重复运行测试）
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"description": "存储设备信息，元数据为布尔类型的示例集合"}
    )
    print(f"集合 {collection_name} 创建成功")

    # 步骤 3：添加数据（核心：元数据5个key均为布尔类型）
    print("\n=== 开始添加带布尔类型元数据的设备数据 ===")
    # 3.1 构造数据（ids、documents、metadatas 长度完全一致）
    device_ids = ["device_001", "device_002", "device_003", "device_004"]
    device_documents = [
        "床边位置感知传感器：支持人体红外感应，靠近自动唤醒，在线运行中",
        "客厅普通吸顶灯：无位置感知功能，离线待机状态，不支持床边使用",
        "飞利浦智能床边灯：具备精准位置定位，在线激活，适配飞利浦生态",
        "卧室普通插座：无位置感知能力，在线运行，无床边设备标识"
    ]
    device_metadatas = [
        # 元数据：5个key均为 bool 类型（True/False）
        {
            "states": True,          # 在线
            "capabilities": True,    # 具备位置感知
            "device_id_clues": True, # 包含床边标识
            "usage_habits": True,    # 符合床边使用习惯
            "others": False          # 不适配飞利浦生态
        },
        {
            "states": False,
            "capabilities": False,
            "device_id_clues": False,
            "usage_habits": False,
            "others": False
        },
        {
            "states": True,
            "capabilities": True,
            "device_id_clues": True,
            "usage_habits": True,
            "others": True
        },
        {
            "states": True,
            "capabilities": False,
            "device_id_clues": False,
            "usage_habits": True,
            "others": False
        }
    ]

    # 3.2 批量添加数据（无报错，bool类型符合稳定版要求）
    collection.add(
        ids=device_ids,
        documents=device_documents,
        metadatas=device_metadatas
    )
    print(f"成功添加 {len(device_ids)} 条设备数据，元数据均为布尔类型")

    # 步骤 4：带布尔条件过滤查询（示例：查找「在线+具备位置感知+符合床边习惯」的设备）
    print("\n=== 执行带布尔条件的语义查询 ===")
    query_text = "床边可用的位置感知设备"
    # 4.1 构造布尔过滤条件（使用官方支持的 $eq 操作符，匹配 bool 值）
    boolean_where_filter = {
        "$and": [  # 多条件且查询，可灵活改为 $or
            {"states": {"$eq": True}},          # 筛选：在线设备
            {"capabilities": {"$eq": True}},    # 筛选：具备位置感知
            {"usage_habits": {"$eq": True}}     # 筛选：符合床边使用习惯
        ]
    }
    print(f"查询文本：{query_text}")
    print(f"过滤条件：{boolean_where_filter}")

    # 4.2 执行查询
    query_results = collection.query(
        query_texts=[query_text],
        where=boolean_where_filter,
        n_results=2,  # 返回前2条最相似结果
        include=["documents", "metadatas", "distances"]  # 包含距离值（相似度）
    )

    # 步骤 5：格式化解析并打印结果
    print("\n=== 查询结果解析（按相似度排序，距离越小越相似）===")
    if not query_results["ids"][0]:
        print("无符合条件的设备结果")
        return

    for idx, (doc_id, doc, meta, dist) in enumerate(
        zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0]
        ),
        1
    ):
        print(f"\n【第 {idx} 条结果】")
        print(f"  设备 ID：{doc_id}")
        print(f"  设备描述：{doc}")
        print(f"  相似度得分（越小越相似）：{dist:.6f}")
        print(f"  布尔类型元数据：")
        print(f"    - 设备是否在线（states）：{meta['states']}")
        print(f"    - 是否具备位置感知（capabilities）：{meta['capabilities']}")
        print(f"    - 是否包含床边标识（device_id_clues）：{meta['device_id_clues']}")
        print(f"    - 是否符合床边习惯（usage_habits）：{meta['usage_habits']}")
        print(f"    - 是否适配飞利浦生态（others）：{meta['others']}")

    return query_results

# 执行示例主流程
if __name__ == "__main__":
    final_results = chromadb_boolean_metadata_demo()