from smartHome.m_agent.agent.persistent_tools import PythonInterpreterTool, NotifyOnConditionTool
from smartHome.m_agent.common.get_llm import get_llm
from langchain.agents import create_agent
from langchain.tools import tool

from smartHome.m_agent.memory.fact_memory import get_device_all_entities_states, get_device_all_entities_capabilities
from smartHome.m_agent.memory.fake.fake_do_service import fake_execute_domain_service_by_entity_id
from smartHome.m_agent.memory.fake.fake_request import fake_get_states_by_entity_id, fake_get_services_by_domain


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
                         tools=[get_device_current_status, execute_device_action, start_device_persistent_monitoring], )
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
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
                         tools=[get_device_all_entities_states,fake_get_states_by_entity_id], )
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
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
                """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_capabilities,fake_get_states_by_entity_id,fake_get_services_by_domain,fake_execute_domain_service_by_entity_id], )
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
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
            3. 编写代码，在代码中可以直接调用函数fake_get_states_by_entity_id()，我会确保其在运行时存在
            4. 调用@tool PythonInterpreterTool 运行一次代码，确保编写执行无误
            5. 调用工具持久化监控
            """
    agent = create_agent(model=get_llm(),
                         tools=[get_device_all_entities_capabilities, fake_get_states_by_entity_id,PythonInterpreterTool,NotifyOnConditionTool], )
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
        ]
    })
    return result["messages"][-1].content