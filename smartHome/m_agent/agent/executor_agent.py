from smartHome.m_agent.agent.langchain_middleware import log_before, log_response, log_before_agent, log_after_agent, \
    AgentContext
from smartHome.m_agent.agent.persistent_tools import PythonInterpreterTool, NotifyOnConditionTool
from smartHome.m_agent.common.get_llm import get_llm
from langchain.agents import create_agent
from langchain.tools import tool

from smartHome.m_agent.common.global_config import GLOBALCONFIG
from smartHome.m_agent.memory.fact_memory import get_device_all_entities_states, get_device_all_entities_capabilities
from smartHome.m_agent.memory.fake.tools_fake_request import tool_get_states_by_entity_id, tool_get_services_by_domain, \
    tool_execute_action_by_entity_id


@tool
def executor_planning(planning:str):
    """
    按照给定的计划表执行计划，返回计划执行结果
    :param planning:
    :return:
    """
    prompt = f"""
        【计划表】:{planning}
        根据计划表，调用不同的工具来完成计划表中的每一个任务，你不需要修正计划表，只需要如实记录各任务执行情况。
        - 如果任务失败，需要简练记录失败原因
        """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_current_status, execute_device_action, start_device_persistent_monitoring],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="执行计划阶段")
    )

    # msg_content = "\n" + "\n".join(map(repr, result["messages"]))
    # GLOBALCONFIG.logger.info("================" + "执行计划")
    # GLOBALCONFIG.logger.info(msg_content)
    # GLOBALCONFIG.logger.info("\n")

    return result["messages"][-1].content

@tool
def get_device_current_status(device_id:str,what_status:str):
    """
    获取设备的某些状态
    :param device_id:
    :param what_status:
    :return:
    """
    prompt = f"""
            【设备ID】：{device_id}
            【任务】：{what_status}
            一个设备有多个实体，你可以首先调用工具查看设备下所有实体各自能得到什么状态类型
            选择实体后再调用工具获取该设备的当前状态
            """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_states,tool_get_states_by_entity_id],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="executor_读设备阶段")
    )
    return result["messages"][-1].content

@tool
def check_device(task:str):
    """
    执行task描述的任务
    :param task:
    :return:
    """
    from smartHome.m_agent.agent.home_agent import run_ourAgent
    return run_ourAgent(task=task)
@tool
def check_smart_home_action_execution(device_id: str,task:str):
    """
    验证某个设备是否执行成功
    :param task:
    :return:
    """
    return "卧室的人体传感器光照度很低，说明灯可能由于硬件原因，实际未打开"
    prompt = f"""
                    【设备ID】：{device_id}
                    【任务】：{task}
                    验证设备是否执行成功，不能通过查看设备自身的状态来检查，而是要利用设备所在的位置的周边的传感器来检查。
                    比如，对于卧室灯164c1a92b8ce9cda0e2a8c13440b4722，我们可以通过卧室的人体传感器53bea65f446cb0a8150250354cb28a40的光照强度来检查出灯实际是否已打开

                    执行动作后，调用工具检查设备是否执行成功，没有成功就如实报告。
                    """
    agent = create_agent(model=get_llm(),
                         tools=[check_device],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="executor_验证设备阶段")
    )
    return result["messages"][-1].content
@tool
def execute_device_action(device_id: str, action: str):
    """
    让设备执行某些操作
    :param device_id:
    :param action:
    :return:
    """
    prompt = f"""
                【设备ID】：{device_id}
                【任务】：{action}
                一个设备有多个实体，你可以首先调用工具查看设备下所有实体各自能进行什么操作
                选择实体后再调用工具对设备执行动作
                
                执行动作后，调用工具检查设备是否执行成功，没有成功就如实报告。
                """
    prompt = f"""
                【设备ID】：{device_id}
                【任务】：{action}
                一个设备有多个实体，你可以首先调用工具查看设备下所有实体各自能进行什么操作
                选择实体后再调用工具对设备执行动作
                """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_capabilities,tool_get_states_by_entity_id,tool_get_services_by_domain,tool_execute_action_by_entity_id,
                                # check_smart_home_action_execution
                                ],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="executor_操作设备阶段")
    )
    return result["messages"][-1].content

@tool
def start_device_persistent_monitoring(device_id: str,when_true:str,then_do:str):
    """
    当某个设备的某种状态满足条件时，执行某些操作
    :param device_id:
    :param when_true:
    :param then_do:
    :return:
    """
    prompt = f"""
            【设备ID】：{device_id}
            【持久化任务】：when_true-{when_true} then_do{then_do}
            1. 一个设备有多个实体，你可以首先调用工具查看设备下所有实体各自能得到什么状态类型
            2. 选择实体后再调用工具获取该实体的json数据，观察其结构组成
            3. 编写代码，在代码中可以直接调用函数tool_get_states_by_entity_id()，我会确保其在运行时存在
            4. 调用@tool PythonInterpreterTool 运行一次代码，确保编写执行无误
            5. 调用工具持久化监控
            """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_capabilities, tool_get_states_by_entity_id,PythonInterpreterTool,NotifyOnConditionTool],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="executor_设备持久监控阶段")
    )
    return result["messages"][-1].content

if __name__ == "__main__":

    device_id="cf03cb835279ea4876ab6ee202aa9832"
    when_true="当门未关闭时"
    then_do="通知我"
    prompt = f"""
                【设备ID】：{device_id}
                【持久化任务】：when_true-{when_true} then_do{then_do}
                1. 一个设备有多个实体，你可以首先调用工具查看设备下所有实体各自能得到什么状态类型
                2. 选择实体后再调用工具获取该实体的json数据，观察其结构组成
                3. 编写代码，在代码中可以直接调用函数tool_get_states_by_entity_id()，我会确保其在运行时存在
                4. 调用@tool PythonInterpreterTool 运行一次代码，确保编写执行无误
                5. 调用工具持久化监控
                """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_capabilities, tool_get_states_by_entity_id,
                                PythonInterpreterTool, NotifyOnConditionTool],
                         middleware=[log_before, log_response, log_before_agent, log_after_agent],
                         context_schema=AgentContext
                         )
    result = agent.invoke(
        input={"messages": [
            {"role": "system", "content": prompt},
        ]},
        context=AgentContext(agent_name="executor_设备持久监控阶段")
    )
    ans=result["messages"][-1].content
    pass