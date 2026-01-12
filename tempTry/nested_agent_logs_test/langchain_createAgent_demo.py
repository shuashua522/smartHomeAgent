from dataclasses import dataclass

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage

from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.common.global_config import GLOBALCONFIG
from langchain.agents.middleware import before_model, after_model, AgentState, before_agent, after_agent
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@dataclass
class AgentContext:
    agent_name: str

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    GLOBALCONFIG.nested_level = GLOBALCONFIG.nested_level + 1
    GLOBALCONFIG.print_nested_log("进入"+runtime.context.agent_name+"======================================")

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    GLOBALCONFIG.nested_level=GLOBALCONFIG.nested_level - 1

@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    message = state['messages'][-1]
    s = repr(message)
    GLOBALCONFIG.print_nested_log(s)
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    message=state['messages'][-1]
    s=repr(message)
    GLOBALCONFIG.print_nested_log(s)
    return None

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"多云"

@tool
def get_weather() -> str:
    """Get weather information for 广州."""
    agent = create_agent(model=get_llm(), tools=[search],middleware=[log_before,log_response,log_before_agent,log_after_agent],context_schema=AgentContext)
    result = agent.invoke({
        "messages": [{"role": "system", "content": "广州天气?"}]},
        context=AgentContext(agent_name="get_weather"))
    return result["messages"][-1].content

agent = create_agent(model=get_llm(), tools=[get_weather],middleware=[log_before,log_response,log_before_agent,log_after_agent],context_schema=AgentContext)
result = agent.invoke(
    {"messages": [{"role": "system", "content": "广州天气怎么样"}]},
    context = AgentContext(agent_name="agent_entry")
)

# # content = [AIMessage(content="广州天气怎么样")]
# msg_content = "\n" + "\n".join(map(repr, result["messages"]))
# GLOBALCONFIG.logger.info(msg_content)
# print(result)
