import mem0
import os

# --------------------------
# 关键配置：先解决网络/API Key问题（国内用户必看）
# --------------------------
# 1. （可选）国内用户配置代理（根据自己的代理端口修改，比如7890/1080等）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 2. 设置OpenAI API Key（替换为你的真实有效Key）
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# --------------------------
# 初始化mem0（1.0.1版本核心方式）
# --------------------------
try:
    memory = mem0.Memory(
        llm={
            "provider": "openai",
            "config": {
                "model": "gpt-3.5-turbo",  # 1.0.1完美兼容该模型
                "temperature": 0.1,
                "max_tokens": 1000,
                "base_url": "https://api.openai.com/v1"  # 代理用户可替换为自己的base_url
            }
        }
    )
except Exception as e:
    print(f"❌ mem0初始化失败：{type(e).__name__} - {str(e)}")
    exit(1)

# --------------------------
# 自定义Schema + 文本提取
# --------------------------
# 定义提取模板（和之前一致）
custom_schema = [
    {
        "name": "order_id",
        "type": "string",
        "description": "客户的订单编号，格式为字母+数字组合（如ORD20251222001）",
        "required": True
    },
    {
        "name": "product_name",
        "type": "string",
        "description": "客户咨询的产品名称，完整且准确",
        "required": True
    },
    {
        "name": "order_amount",
        "type": "float",
        "description": "订单金额，保留2位小数的数字",
        "required": True
    },
    {
        "name": "complaint_reason",
        "type": "list",
        "description": "客户投诉的原因，多个原因用列表表示（如['物流慢', '产品破损']）",
        "required": False
    },
    {
        "name": "is_return_request",
        "type": "boolean",
        "description": "客户是否提出退货申请，是则为True，否则为False",
        "required": True
    }
]

# 待提取文本
raw_text = """
客户反馈：我的订单ORD20251222001购买的苹果15 Pro Max手机有问题，订单金额是8999.99元。
一方面物流延迟了3天，另一方面手机屏幕有划痕，我希望能退货。
"""

# 执行提取
try:
    structured_data = memory.extract(
        text=raw_text,
        schema=custom_schema,
        instructions="严格按照Schema提取，金额保留2位小数，投诉原因拆分为列表"
    )
    # 打印结果
    print("✅ 提取成功！结构化信息：")
    for k, v in structured_data.items():
        print(f"  - {k}: {v} (类型: {type(v).__name__})")
except Exception as e:
    print(f"❌ 提取结构化信息失败：{type(e).__name__} - {str(e)}")