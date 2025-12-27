from typing import TypedDict, Literal

from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict, Annotated
import operator

from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.memory.fact_memory import SMARTHOMEMEMORY

from pydantic import BaseModel, Field
from typing import List  # 推荐导入List，规范类型注解

class DeviceIdList(BaseModel):
    """多个智能家居设备的事实性信息列表模型"""
    device_ids: List[str] = Field(  # 替换list[str]为List[str]（更规范）
        default=[],
        description="所有候选设备的ID",
        examples=[["31ae92d8a163d77f8d6a5741c0d1b89c","31ae92d8a163d77f8d6a54856d1b89c"]]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                # 核心修正：用字典匹配模型字段结构
                {
                    "device_ids": ["31ae92d8a163d77f8d6a5741c0d1b89c","31ae92d8a163d77f8d6a54856d1b89c"]
                }
            ]
        }
    }

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
    task=state["command"]
    device_fact_dict=SMARTHOMEMEMORY.device_fact
    device_fact_str_list=[]
    for device_id,device_fact in device_fact_dict.items():
        device_name=device_fact["device_name"]
        # 将device_fact中的states（字符串列表）拼接成一个字符串，逗号分隔
        states_list = device_fact.get("states", [])
        states_str = ", ".join(states_list) if states_list else "无"
        # 将device_fact中的capabilities（字符串列表）拼接成一个字符串，逗号分隔
        capabilities_list = device_fact.get("capabilities", [])
        capabilities_str = ", ".join(capabilities_list) if capabilities_list else "无"
        # 将device_id和states字符串、capabilities字符串拼接形成device_brief
        # 格式示例："dev_001 | 状态：开关状态、亮度 | 能力：打开灯光、调节色温"
        a_device_brief = f"{device_id}({device_name}) | 可获取状态：{states_str} | 能力：{capabilities_str}"
        device_fact_str_list.append(a_device_brief)
    devices_brief = "\n".join(device_fact_str_list) if device_fact_str_list else "无"

    prompt=f"""
    根据所有的设备简介（设备能做什么，能从设备获取到什么），选出可能能用于完成此次任务的设备ID列表
    【任务】
    {task}
    【设备简要】
    {devices_brief}
    """
    agent = create_agent(
        model=get_llm(),
        response_format=DeviceIdList,
    )
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    deviceIdList = result["structured_response"]
    device_ids=deviceIdList.device_ids
    device_ids_str=",".join(device_ids)
    content = [AIMessage(content=device_ids_str)]
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

    second_filter_prompt="""
    如果用户指令包含约束条件，根据约束条件（设备环境信息、用户对设备的称呼等），从候选设备里挑出满足的设备
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
    根据设备能力和获取状态、当前状态、使用习惯，规划出应该让每个设备做什么
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

