from typing import Union, Dict, List

from langchain.tools import tool

@tool
def fake_get_services_by_domain(domain:str):
    """
    获取服务
    :param domain:
    :return:
    """
    pass
@tool
def fake_get_all_entities():
    """
    获取所有实体
    :return:
    """
    pass
@tool
def fake_get_states_by_entity_id(entity_id):
    """
    获取某个实体的状态
    :param entity_id:
    :return:
    """
    pass

@tool
def fake_execute_domain_service_by_entity_id(domain, service, body, ) -> Union[Dict, List]:
    """
    执行操作
    :param domain:
    :param service:
    :param body:
    :return:
    """
    pass