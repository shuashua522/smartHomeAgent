from dataclasses import dataclass
from smartHome.m_agent.common.global_config import GLOBALCONFIG
from langchain.agents.middleware import before_model, after_model, AgentState, before_agent, after_agent
from langgraph.runtime import Runtime
from typing import Any

@dataclass
class AgentContext:
    agent_name: str

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    GLOBALCONFIG.add_agent_name(runtime.context.agent_name)
    GLOBALCONFIG.print_nested_log("进入 "+runtime.context.agent_name+" ======================================")

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    GLOBALCONFIG.delete_agent_name(runtime.context.agent_name)

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