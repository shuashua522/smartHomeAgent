from langchain.tools import tool
from langchain.agents import create_agent

from smartHome.m_agent.common.get_llm import get_llm


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model=get_llm(), tools=[search, get_weather])
result = agent.invoke({
    "messages": [{"role": "system", "content": "广州天气怎么样"}]
})
print(result)