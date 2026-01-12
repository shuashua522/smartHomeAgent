from typing import TypedDict, Literal

from langchain.agents import create_agent

from langgraph.types import Command
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict, Annotated
import operator

from smartHome.m_agent.agent.executor_agent import executor_planning
from smartHome.m_agent.agent.langchain_middleware import AgentContext, log_before, log_response, log_before_agent, \
    log_after_agent
from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.common.global_config import GLOBALCONFIG
from smartHome.m_agent.memory.fact_memory import SMARTHOMEMEMORY

from pydantic import BaseModel, Field
from typing import List  # 推荐导入List，规范类型注解

from smartHome.m_agent.memory.vector_device import get_device_constraints_individual_match_text, \
    get_device_all_states, get_device_all_capabilities, get_device_all_usage_habits, get_devices_states, \
    get_devices_capabilities, get_devices_usage_habits


class DeviceInfo(BaseModel):
    """单个智能家居设备的完整信息模型（包含ID、名称、选择理由）"""
    device_id: str = Field(
        description="设备唯一标识ID",
        examples=["31ae92d8a163d77f8d6a5741c0d1b89c"]
    )
    device_name: str = Field(
        description="设备名称",
        examples=["客厅智能吸顶灯"]
    )
    device_reason: str = Field(
        description="选择该设备的理由（50字以内）",
        examples=["亮度可调节，能匹配客厅日常照明和观影场景需求"]
    )

class DeviceIdList(BaseModel):
    """多个智能家居设备的事实性信息列表模型"""
    devices: List[DeviceInfo] = Field(
        default=[],
        description="所有候选设备的完整信息列表（ID、名称、选择理由）",
        examples=[
            [
                {
                    "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",
                    "device_name": "客厅智能吸顶灯",
                    "device_reason": "亮度可调节，能匹配客厅日常照明和观影场景需求"
                },
                {
                    "device_id": "31ae92d8a163d77f8d6a54856d1b89c",
                    "device_name": "卧室智能窗帘",
                    "device_reason": "支持定时开合，能配合作息自动调节卧室采光"
                }
            ]
        ]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "devices": [
                        {
                            "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",
                            "device_name": "客厅智能吸顶灯",
                            "device_reason": "亮度可调节，能匹配客厅日常照明和观影场景需求"
                        },
                        {
                            "device_id": "31ae92d8a163d77f8d6a54856d1b89c",
                            "device_name": "卧室智能窗帘",
                            "device_reason": "支持定时开合，能配合作息自动调节卧室采光"
                        }
                    ]
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

    first_filter_devices: str
    second_filter_devices: str

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
    json_str_compact = """{"devices":[{"device_id":"c86e3c14d0egbfc02g4cae35662d6944","device_name":"灯泡 灯","device_reason":"支持开关控制，可关闭"},{"device_id":"164c1a92b8ce9cda0e2a8c13440b4722","device_name":"灯泡  灯","device_reason":"支持开关控制，可关闭"},{"device_id":"b75d2b03c9dfaebf1f3b9d24551c5833","device_name":"灯泡  灯","device_reason":"支持开关控制，可关闭"},{"device_id":"31ae92d8a163d77f8d6a5741c0d1b89c","device_name":"米家智能台灯Lite","device_reason":"支持开关控制，可关闭"},{"device_id":"e2bf03e9b274e88f9e7b6852d1e2c90d","device_name":"米家智能台灯Lite","device_reason":"支持开关控制，可关闭"}]}"""
    content = [AIMessage(content=json_str_compact)]
    return Command(
        update={"messages": content,
                "first_filter_devices": json_str_compact},  # Store raw results or error
        goto="filter_2_node"
    )

    system_prompt = f"""
        根据所有的设备简介（设备能做什么，能从设备获取到什么），选出可能能用于完成此次任务的设备ID列表，并简单说明理由.
        - 如果【任务】里的设备有限定条件，比如床边的灯、卧室的空调等，你不需要在意限定条件，只需要根据设备能做什么、能获取什么来筛选可能的设备，之后会根据限定条件进一步筛选
        """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_states, get_device_all_capabilities],
                        system_prompt=system_prompt,
                         response_format=DeviceIdList,
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema = AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "user", "content": state['command']},
        ]},
        context=AgentContext(agent_name="过滤一")
    )
    # # 日志打印
    # msg_content = "\n" + "\n".join(map(repr, result["messages"]))
    # GLOBALCONFIG.logger.info("================"+"过滤一")
    # GLOBALCONFIG.logger.info(msg_content)
    # GLOBALCONFIG.logger.info("\n")

    deviceInfoList = result["structured_response"]
    # 无缩进（紧凑格式，适合传输/存储）
    json_str_compact = deviceInfoList.model_dump_json()
    # 带缩进（美化格式，适合调试/查看）
    json_str_pretty = deviceInfoList.model_dump_json(indent=4)

    content = [AIMessage(content=json_str_compact)]
    return Command(
        update={"messages": content,
                "first_filter_devices":json_str_compact},  # Store raw results or error
        goto="filter_2_node"
    )

def node_filter_2(state:SmartHomeAgentState)-> Command[Literal["planner_node", END]]:
    """
    筛选设备
    :param state:
    :return:
    """
    # 不需要工具，从记忆里获取信息，进行初筛 和 细筛
    json_str_compact = """{"devices":[{"device_id":"c86e3c14d0egbfc02g4cae35662d6944","device_name":"灯泡 灯","device_reason":"支持开关控制，可关闭"},{"device_id":"164c1a92b8ce9cda0e2a8c13440b4722","device_name":"灯泡  灯","device_reason":"支持开关控制，可关闭"}]}"""

    content = [AIMessage(content=json_str_compact)]
    return Command(
        update={"messages": content,
                "second_filter_devices": json_str_compact},  # Store raw results or error
        goto="planner_node"
    )

    prompt=f"""
    【任务】：{state["command"]}
    【候选设备集】：{state["first_filter_devices"]}
    如果用户指令包含约束条件，根据约束条件（设备环境信息、用户对设备的称呼等），从候选设备里挑出满足的设备。
    - 比如用户要打开客厅餐桌的灯和卧室床边的灯，候选设备集里有两盏灯（灯1和灯2），那么需要调用tool获取这两盏灯各自与[[客厅，餐桌],[卧室，床边]]的相似记忆内容。
    - 然后检查是否有说明这两盏灯满足条件，如果都不满足，那么不应该选用。比如灯1的检索到的记忆既没说明其在卧室，也没说明其在客厅，那么不该选出灯1
    最后保留设备ID，和简单说明理由。如果没有任何设备满足约束条件，说明原因。
    """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_constraints_individual_match_text],
                         # response_format=DeviceIdList,
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context = AgentContext(agent_name="过滤二")
    )
    # # 日志打印
    # msg_content = "\n" + "\n".join(map(repr, result["messages"]))
    # GLOBALCONFIG.logger.info("================" + "过滤二")
    # GLOBALCONFIG.logger.info(msg_content)
    # GLOBALCONFIG.logger.info("\n")

    # deviceInfoList = result["structured_response"]
    # # 无缩进（紧凑格式，适合传输/存储）
    # json_str_compact = deviceInfoList.model_dump_json()
    # # 带缩进（美化格式，适合调试/查看）
    # json_str_pretty = deviceInfoList.model_dump_json(indent=4)
    json_str_compact=result["messages"][-1].content

    content = [AIMessage(content=json_str_compact)]
    return Command(
        update={"messages": content,
                "second_filter_devices": json_str_compact},  # Store raw results or error
        goto="planner_node"
    )

def node_planner(state:SmartHomeAgentState)-> Command[Literal["deliver_node", END]]:
    """
    规划和执行
    :param state:
    :return:
    """
    prompt=f"""
    【任务】：{state["command"]}
    【候选设备集】：{state["second_filter_devices"]}
    根据任务及设备能力和状态类型、使用习惯，规划出应该让每个设备做什么
    """
    # todo 改成从设备列表获取 设备能力和获取状态
    agent = create_agent(
        model=get_llm(),
        tools=[get_devices_states,get_devices_capabilities,get_devices_usage_habits,executor_planning],
        middleware=[log_before, log_response, log_before_agent, log_after_agent],
        context_schema=AgentContext
    )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context = AgentContext(agent_name="规划阶段")
    )

    # msg_content = "\n" + "\n".join(map(repr, result["messages"]))
    # GLOBALCONFIG.logger.info("================" + "规划阶段")
    # GLOBALCONFIG.logger.info(msg_content)
    # GLOBALCONFIG.logger.info("\n")

    content = [AIMessage(content=result["messages"][-1].content)]
    return Command(
        update={"messages": content},  # Store raw results or error
        goto="deliver_node"
    )


def node_deliver(state:SmartHomeAgentState)-> Command[Literal[END]]:
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
agent_builder.add_node("deliver_node", node_deliver)

agent_builder.add_edge(START, "filter_1_node")
# agent_builder.add_edge("filter_node", "planner_and_executor_node")
# agent_builder.add_edge("planner_and_executor_node", END)

agent = agent_builder.compile()

# Test with an urgent billing issue
task="关闭所有灯"
initial_state = {
    "command": task,
    "messages": [HumanMessage(content=task)]
}
result = agent.invoke(initial_state)
for m in result["messages"]:
    m.pretty_print()

