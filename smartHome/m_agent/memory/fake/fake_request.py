import json
import os
from typing import Union, Dict, List



# 获取当前 Python 文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 拼接 entities.json 和 services.json 的绝对路径
entities_path = os.path.join(parent_dir, "copied_data","entities.json")
services_path = os.path.join(parent_dir, "copied_data","domains_services.json")

# 读取 JSON 文件
entities = json.load(open(entities_path, "r", encoding="utf-8"))
services = json.load(open(services_path, "r", encoding="utf-8"))


def fake_get_services_by_domain(domain:str):
    """
    获取domain下的所有服务
    :param domain:
    :return:
    """
    for service in services:
        if domain == service["domain"]:
            return service
    return None

def fake_get_all_entities():
    """
    获取所有实体
    :return:
    """
    return entities

def fake_get_states_by_entity_id(entity_id):
    """
    获取实体当前的json数据
    :param entity_id:
    :return:
    """
    for entity in entities:
        if entity.get("entity_id") == entity_id:  # 使用get避免KeyError
            return entity
    return None

