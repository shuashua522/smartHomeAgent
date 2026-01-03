from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models.Collection import Collection


# 1. 数据模型（保持不变）
class TextWithMeta(BaseModel):
    """单条文本的数据模型（含内容、多标签、元信息）"""
    text_id: str  # 文本唯一标识（方便后续更新/删除）
    content: str  # 核心文本内容（用于生成向量）
    tags: List[str]  # 多标签（支持检索过滤，如["智能家居", "设备手册"]）
    create_time: datetime  # 创建时间（元信息）
    update_time: Optional[datetime] = None  # 更新时间（元信息，可选）
    other_meta: Optional[dict] = None  # 其他自定义元信息（如作者、来源等）


# 2. 初始化文本嵌入函数（保持不变）
embedding_func = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # 轻量高效，支持中英文
)

# 3. 初始化Chroma向量数据库（保持不变，支持持久化）
client = chromadb.PersistentClient(path="./chroma_text_db")


# 4. 创建/获取集合（保持不变）
def get_or_create_collection(collection_name: str = "texts_with_tags_and_meta") -> Collection:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"description": "存储文本，关联多标签和创建/更新时间等元信息"}
    )


# 初始化集合
text_collection = get_or_create_collection()


# 5. 修正后的入库函数（保持不变，None替换为合法字符串）
def add_text_to_vector_db(text_data: TextWithMeta, collection: Collection):
    """将单条文本（含标签、元信息）存入向量数据库，tags列表拆分为独立字段"""
    # 1. 处理自定义元信息，避免内部包含None值
    other_meta = text_data.other_meta or {}
    cleaned_other_meta = {k: v if v is not None else "N/A" for k, v in other_meta.items()}

    # 2. 初始化元数据字典
    metadata = {
        "create_time": text_data.create_time.isoformat(),
        "update_time": text_data.update_time.isoformat() if text_data.update_time else "未更新",
        **cleaned_other_meta
    }

    # 3. 遍历tags列表，拆分为多个独立Metadata字段
    for idx, tag in enumerate(text_data.tags):
        metadata[f"tag_{idx}"] = tag

    # 4. 入库操作
    collection.add(
        ids=[text_data.text_id],
        documents=[text_data.content],
        metadatas=[metadata]
    )
    print(f"✅ 文本「{text_data.text_id}」已成功存入向量数据库")


# 6. 新增：从metadata中还原tags列表（复用之前的函数）
def restore_tags_from_metadata(metadata: dict) -> List[str]:
    """从拆分后的metadata中还原原始tags列表"""
    # 筛选所有tag_*开头的字段，按索引排序
    tag_fields = sorted(
        [k for k in metadata.keys() if k.startswith("tag_")],
        key=lambda x: int(x.split("_")[1])  # 按索引数字升序排列
    )
    # 提取标签值并返回
    return [metadata[field] for field in tag_fields]


# 7. 新增：核心检索函数（支持相似性检索+元数据过滤）
def search_text_in_vector_db(
        query_text: str,
        collection: Collection,
        where_filter: Optional[dict] = None,
        n_results: int = 2  # 返回最相似的n条结果
) -> dict:
    """
    从向量数据库中检索相似文本
    :param query_text: 查询文本（用于生成向量，匹配相似内容）
    :param collection: ChromaDB集合
    :param where_filter: 元数据过滤条件（ChromaDB查询语法），可选
    :param n_results: 返回结果数量
    :return: 整理后的检索结果（含还原的tags、完整元信息）
    """
    # 1. 执行检索（Chroma自动为query_text生成向量，进行相似性匹配）
    results = collection.query(
        query_texts=[query_text],  # 查询文本列表（单条查询传入长度为1的列表）
        where=where_filter,  # 元数据过滤条件（如{"author": "admin"}）
        n_results=n_results,  # 返回最相似的n条结果
        include=["documents", "metadatas", "distances"]  # 指定返回的内容（文档、元数据、相似度距离）
    )

    # 2. 整理检索结果，还原tags列表，提升可读性
    cleaned_results = {
        "query_text": query_text,
        "total_matches": len(results["ids"][0]),
        "matches": []
    }

    for idx, (text_id, document, metadata, distance) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        # 还原原始tags列表
        original_tags = restore_tags_from_metadata(metadata)

        # 整理单条匹配结果
        match_item = {
            "rank": idx + 1,  # 匹配排名（1为最相似）
            "text_id": text_id,
            "content": document,
            "tags": original_tags,  # 还原后的标签列表
            "metadata": {
                "create_time": metadata.get("create_time", "N/A"),
                "update_time": metadata.get("update_time", "N/A"),
                "author": metadata.get("author", "N/A"),
                "source": metadata.get("source", "N/A")
            },
            "similarity_distance": round(distance, 4)  # 相似度距离（值越小，相似度越高）
        }
        cleaned_results["matches"].append(match_item)

    return cleaned_results


# 8. 新增：打印格式化检索结果（方便查看）
def print_search_results(search_results: dict):
    """格式化打印检索结果"""
    print("\n" + "=" * 80)
    print(f"🔍 检索查询：{search_results['query_text']}")
    print(f"📊 匹配结果数量：{search_results['total_matches']}")
    print("=" * 80)

    for match in search_results["matches"]:
        print(f"\n🏆 排名：{match['rank']}")
        print(f"📄 文本ID：{match['text_id']}")
        print(f"📝 文本内容：{match['content']}")
        print(f"🏷️  标签：{match['tags']}")
        print(f"📋 元信息：")
        for k, v in match["metadata"].items():
            print(f"  - {k}：{v}")
        print(f"📈 相似度距离（越小越相似）：{match['similarity_distance']}")
        print("-" * 50)


# 9. 测试流程：入库 + 两次检索验证
if __name__ == "__main__":
    # 第一步：构造测试数据并入库（保持不变）
    print("===== 开始入库测试数据 =====")
    test_text1 = TextWithMeta(
        text_id="text_001",
        content="客厅智能吸顶灯支持亮度调节和色温切换，可通过手机APP远程控制。",
        tags=["智能家居", "照明设备", "客厅"],
        create_time=datetime.now(),
        other_meta={"author": "admin", "source": "设备使用手册"}
    )
    add_text_to_vector_db(test_text1, text_collection)

    test_text2 = TextWithMeta(
        text_id="text_002",
        content="卧室智能窗帘支持定时开合，配合作息自动调节卧室采光。",
        tags=["智能家居", "窗帘设备", "卧室"],
        create_time=datetime.now(),
        update_time=datetime.now(),
        other_meta={"author": "admin", "source": "设备使用手册"}
    )
    add_text_to_vector_db(test_text2, text_collection)

    # 第二步：检索测试1 - 普通相似性检索（无过滤，匹配所有相关文本）
    print("\n===== 开始检索测试1：普通相似性检索 =====")
    query1 = "客厅照明设备如何控制？"  # 贴近text_001的查询
    search_results1 = search_text_in_vector_db(
        query_text=query1,
        collection=text_collection,
        n_results=2
    )
    print_search_results(search_results1)

    # 第三步：检索测试2 - 带元数据过滤的检索（仅匹配"智能家居"标签+作者admin）
    print("\n===== 开始检索测试2：带过滤条件的检索 =====")
    query2 = "智能家居设备使用说明"  # 通用查询
    # 过滤条件：tag_0="智能家居"（第一个标签）且 author="admin"（ChromaDB支持$eq/$in等语法）
    where_filter = {
        "$and": [
            {"tag_0": "智能家居"},
            {"author": "admin"}
        ]
    }
    search_results2 = search_text_in_vector_db(
        query_text=query2,
        collection=text_collection,
        where_filter=where_filter,
        n_results=2
    )
    print_search_results(search_results2)