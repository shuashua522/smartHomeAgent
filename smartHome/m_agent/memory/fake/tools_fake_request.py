from typing import Union, Dict, List

from langchain.tools import tool

from smartHome.m_agent.memory.fake.fake_do_service import fake_execute_domain_service_by_entity_id
from smartHome.m_agent.memory.fake.fake_request import fake_get_services_by_domain, fake_get_all_entities, \
    fake_get_states_by_entity_id


@tool
def tool_get_services_by_domain(domain:str):
    """
    获取domain下的所有服务
    :param domain:
    :return:
    """
    return fake_get_services_by_domain(domain)
@tool
def tool_get_all_entities():
    """
    获取所有实体
    :return:
    """
    return fake_get_all_entities()
@tool
def tool_get_states_by_entity_id(entity_id:str):
    """
    获取实体当前的json数据
    :param entity_id:
    :return:
    """
    return fake_get_states_by_entity_id(entity_id)

@tool
def tool_execute_action_by_entity_id(domain:str, service:str, body:str ) -> Union[Dict, List]:
    """
    执行操作：
    Calls a service within a specific domain. Will return when the service has been executed.

    Returns a list of states that have changed while the service was being executed, and optionally response data, if supported by the service.
    :param domain: entity_id的前缀即为对应的domain，比如某一entity_id为switch.cuco_cn_269067598_cp1_on_p_2_1，其domain即为switch
    :param service: 通过调用工具@get_services_by_domain获取对应domain下的所有的services，从中选择需要执行的服务
    :param body:'Content-Type': 'application/json'。请求体至少包含'entity_id'(body中有且仅能出现一个entity_id)，如果service还需要其他的参数，请补足。通过调用工具@get_all_entity_id可以获取所有的entity_id，从中选择所需的entity_id进行操作。
    :return:
    """
    return fake_execute_domain_service_by_entity_id(domain, service, body)