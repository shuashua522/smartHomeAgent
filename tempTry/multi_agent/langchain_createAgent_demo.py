from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage

from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.common.global_config import GLOBALCONFIG


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather() -> str:
    """Get weather information for 广州."""
    location="广州"
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model=get_llm(), tools=[search, get_weather])
result = agent.invoke({
    "messages": [{"role": "system", "content": "广州天气怎么样"}]
})

# # content = [AIMessage(content="广州天气怎么样")]
msg_content = "\n" + "\n".join(map(repr, result["messages"]))
GLOBALCONFIG.logger.info(msg_content)
# print(result)

# 创建一个LangChain SystemMessage实例
sys_msg = SystemMessage(
    content="广州天气怎么样",
    additional_kwargs={},
    id='368e8b9b-0107-4481-8990-7015f192bdd1'
)

# 1. 调用str()（对应你的日志打印逻辑）
print("=== 调用str()（日志打印的单个元素）===")
print(str(sys_msg))

# 2. 调用repr()（对应print(result)中列表元素的处理逻辑）
print("\n=== 调用repr()（直接print(result)的单个元素）===")
print(repr(sys_msg))