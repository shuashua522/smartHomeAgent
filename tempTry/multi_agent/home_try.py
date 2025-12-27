from typing import TypedDict, Literal
from langgraph.types import Command
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict, Annotated
import operator

# Define the structure for email classification
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class SmartHomeAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    command: str
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_results: list[str] | None  # List of raw document chunks
    customer_history: dict | None  # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    # messages: list[str] | None

    llm_calls: int


def node_filter_1(state:SmartHomeAgentState)-> Command[Literal["filter_2_node", END]]:
    """
    筛选设备
    :param state:
    :return:
    """
    # 不需要工具，从记忆里获取信息，进行初筛 和 细筛
    first_filter_prompt="""
    根据所有的设备简介（设备能做什么，能从设备获取到什么），选出可能能用于完成此次任务的设备列表
    """
    result=["device_01","device_02"]
    content = [AIMessage(content="device_01,device_02")]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto="filter_2_node"
    )

def node_filter_2(state:SmartHomeAgentState)-> Command[Literal["planner_node", END]]:
    """
    筛选设备
    :param state:
    :return:
    """
    # 不需要工具，从记忆里获取信息，进行初筛 和 细筛
    first_filter_prompt="""
    根据所有的设备简介（设备能做什么，能从设备获取到什么），选出可能能用于完成此次任务的设备列表
    """
    second_filter_prompt="""
    根据约束条件（设备环境信息、用户对设备的称呼等），从候选设备里挑出满足的设备
    """
    content = [AIMessage(content="device_01")]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto="planner_node"
    )

def node_planner(state:SmartHomeAgentState)-> Command[Literal["executor_node", END]]:
    """
    规划和执行
    :param state:
    :return:
    """
    planner_prompt="""
    根据设备能力和获取状态，规划出应该让每个设备做什么
    """
    executor_prompt="""
    三个执行体，根据计划表，选用相应的执行体完成任务
    """
    content = [AIMessage(content="对device_01开灯")]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto="executor_node"
    )

def node_executor(state:SmartHomeAgentState)-> Command[Literal["deliver_node", END]]:
    content = [AIMessage(content="使用device_01的实体light开灯完毕")]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto="deliver_node"
    )

def node_deliver(state:SmartHomeAgentState)-> Command[Literal["deliver_node", END]]:
    """
    交付任务
    :param state:
    :return:
    """
    content = [AIMessage(content="已开灯")]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto=END
    )

agent_builder = StateGraph(SmartHomeAgentState)
# Add nodes
agent_builder.add_node("filter_1_node", node_filter_1)
agent_builder.add_node("filter_2_node", node_filter_2)
agent_builder.add_node("planner_node", node_planner)
agent_builder.add_node("executor_node", node_executor)
agent_builder.add_node("deliver_node", node_deliver)

agent_builder.add_edge(START, "filter_1_node")
# agent_builder.add_edge("filter_node", "planner_and_executor_node")
# agent_builder.add_edge("planner_and_executor_node", END)

agent = agent_builder.compile()

# Test with an urgent billing issue
initial_state = {
    "command": "开床边灯",
    "messages": [HumanMessage(content="开床边灯")]
}
result = agent.invoke(initial_state)
for m in result["messages"]:
    m.pretty_print()

