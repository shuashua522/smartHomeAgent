from pydantic import BaseModel, Field
from langchain.agents import create_agent

from smartHome.m_agent.common.get_llm import get_llm


# 1. 定义Pydantic模型：适配Pydantic V2语法，移除Field的example参数，改用json_schema_extra
class UserLikes(BaseModel):
    """Fact-based information about what the user likes (stored as a list of strings)"""
    likes: list[str] = Field(
        description="事实性的用户喜欢的事物，每个喜欢的内容独立为一个简短字符串，仅保留客观事实，不包含主观评价、修饰词或解释性内容",
    )

    # Pydantic V2 正确添加示例的方式：通过json_schema_extra
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"likes": ["coffee", "hiking", "sci-fi movies"]}
            ]
        }
    }


# 2. 创建Agent（替换为实际可用的模型，如gpt-4o）
agent = create_agent(
    model=get_llm(),  # 需确保已配置OPENAI_API_KEY环境变量
    response_format=UserLikes,
)

# 3. 对话记录（保持不变）
dialogue_record = """
用户：我平时周末喜欢去公园爬山，还喜欢喝手冲咖啡，最近迷上了看科幻电影。
客服：您喜欢的爬山是去近郊的山吗？
用户：对，比如香山，而且我也喜欢吃草莓味的冰淇淋，不喜欢巧克力的。
"""

# 4. 优化提取指令：明确要求“简短字符串”“仅保留核心事实”
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""从以下对话记录中提取用户喜欢的事实性信息，严格遵守：
1. 仅提取“喜欢”的内容，忽略“不喜欢”的内容；
2. 每个内容简化为简短字符串（如“爬山”而非“周末去公园爬山”）；
3. 去重，核心事实保留一个（如“香山爬山”合并为“爬山”）；
4. 仅输出字符串列表，无额外解释。

对话记录：
{dialogue_record}"""
        }
    ]
})

# 5. 输出结果
extracted_likes = result["structured_response"].likes
print("用户喜欢的事实性信息（简洁版）：")
print(extracted_likes)