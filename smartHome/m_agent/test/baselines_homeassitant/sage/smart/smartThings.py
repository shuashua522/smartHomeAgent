import json
import os
from typing import Annotated, List, Callable
from langchain_core.tools import tool
import requests
from typing import Dict, List, Union  # Union 用于表示"或"关系
from langchain.chat_models import init_chat_model
import configparser
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
import json
from typing import Dict, Optional, List

from agent_project.agentcore.commons.base_agent import BaseToolAgent
from agent_project.agentcore.commons.utils import get_llm
from agent_project.agentcore.config.global_config import HOMEASSITANT_AUTHORIZATION_TOKEN, HOMEASSITANT_SERVER, \
    ACTIVE_PROJECT_ENV, PRIVACYHANDLER
from agent_project.agentcore.smart_home_agent.fake_request.fake_do_service import \
    fake_execute_domain_service_by_entity_id
from agent_project.agentcore.smart_home_agent.fake_request.fake_get_entity import fake_get_all_entities, \
    fake_get_services_by_domain, fake_get_states_by_entity_id
from agent_project.agentcore.smart_home_agent.privacy_handler import RequestBodyDecodeAgent, replace_encoded_text, \
    jsonBodyDecodeAndCalc
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.device_doc import \
    Device_info_doc

token = HOMEASSITANT_AUTHORIZATION_TOKEN
server = HOMEASSITANT_SERVER
active_project_env = ACTIVE_PROJECT_ENV
privacyHandler = PRIVACYHANDLER

# 模拟数据的文件所在目录
mock_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # 当前文件所在目录
    'test_mock_data'  # 子目录（无开头斜杠）
)


def extract_entity_by_id(json_file_path: str, target_entity_id: str) -> Optional[Dict]:
    """
    从指定JSON文件中提取目标entity_id对应的字典数据

    参数:
        json_file_path: JSON文件的完整路径（如"D:/homeassistant/entities.json"）
        target_entity_id: 要提取的实体ID（如"sun.sun"、"person.shua"）

    返回:
        若找到目标entity_id，返回对应的字典；若未找到或出现异常，返回None
    """
    # 初始化结果为None，默认未找到目标
    target_entity: Optional[Dict] = None

    # 1. 读取JSON文件内容
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # 2. 解析JSON数据，转换为Python列表（数据结构为List[Dict]）
        entity_list: List[Dict] = json.load(f)

        # 3. 验证解析后的数据是否为列表（防止JSON文件格式错误）
        if not isinstance(entity_list, list):
            print(f"错误：JSON文件内容不是列表格式，实际类型为{type(entity_list).__name__}")
            return None

        # 4. 遍历列表，匹配目标entity_id
        for entity in entity_list:
            # 检查当前字典是否包含"entity_id"键（防止数据不完整）
            if "entity_id" not in entity:
                print(f"警告：跳过无效数据，该字典缺少'entity_id'键：{entity}")
                continue

            # 匹配到目标entity_id，记录结果并退出循环
            if entity["entity_id"] == target_entity_id:
                target_entity = entity
                break
    return target_entity

@tool
def get_all_entity_id() -> Union[Dict, List]:
    """
    Returns an array of state objects.
    Each state has the following attributes: entity_id, state, last_changed and attributes.
    """
    result = None
    if active_project_env == "pro":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"http://{server}/api/states"

        # 发送GET请求
        response = requests.get(url, headers=headers)
        # 检查请求是否成功
        response.raise_for_status()
        # 返回JSON响应内容
        result = response.json()
    elif active_project_env == "dev":
        result=fake_get_all_entities()
    elif active_project_env == "test":
        file_path = os.path.join(mock_data_dir, 'selected_entities.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            # 解析JSON文件并返回Python对象
            result = json.load(f)

    return result


@tool
def get_services_by_domain(domain) -> Union[Dict, List]:
    """
    return all services included in the domain.
    """
    if active_project_env == "pro":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"http://{server}/api/services"

        # 发送GET请求
        response = requests.get(url, headers=headers)
        # 检查请求是否成功
        response.raise_for_status()
        # 返回JSON响应内容
        all_domain_and_services = response.json()
        for domain_entry in all_domain_and_services:
            # 匹配目标 domain
            if domain_entry.get("domain") == domain:
                # 返回该 domain 对应的 services 字典
                return domain_entry
            # 若未找到目标 domain，返回空字典
        return {}
    elif active_project_env == "dev":
        return fake_get_services_by_domain(domain)


@tool
def get_states_by_entity_id(entity_id: Annotated[str, "check the status of {entity_id}"], ) -> Union[Dict, List]:
    """
    Returns a state object for specified entity_id.
    Returns 404 if not found.
    """

    result = None
    if active_project_env == "pro":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"http://{server}/api/states/{entity_id}"

        # 发送GET请求
        response = requests.get(url, headers=headers)
        # 检查请求是否成功
        response.raise_for_status()
        # 返回JSON响应内容
        result = response.json()
    elif active_project_env == "dev":
        result=fake_get_states_by_entity_id(entity_id)
    elif active_project_env == "test":
        file_path = os.path.join(mock_data_dir, 'selected_entities.json')
        result = extract_entity_by_id(file_path, entity_id)

    return result


@tool
def execute_domain_service_by_entity_id(
        domain: Annotated[
            str, "entity_id的前缀即为对应的domain，比如某一entity_id为switch.cuco_cn_269067598_cp1_on_p_2_1，其domain即为switch"],
        service: Annotated[
            str, "通过调用工具@get_services_by_domain获取对应domain下的所有的services，从中选择需要执行的服务"],
        body: Annotated[str, """'Content-Type': 'application/json'。请求体至少包含'entity_id'(body中有且仅能出现一个entity_id)，如果service还需要其他的参数，请补足。
                             通过调用工具@get_all_entity_id可以获取所有的entity_id，从中选择所需的entity_id进行操作。"""],
) -> Union[Dict, List]:
    """
    Calls a service within a specific domain. Will return when the service has been executed.

    Returns a list of states that have changed while the service was being executed, and optionally response data, if supported by the service.
    """
    import agent_project.agentcore.config.global_config as global_config
    logger = global_config.GLOBAL_AGENT_DETAILED_LOGGER
    if logger != None:
        logger.info("\n请求的body:\n" + body)
    result = None
    if active_project_env == "pro":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"http://{server}/api/services/{domain}/{service}"
        # 设置请求体数据
        # payload = {
        #     "entity_id": entity_id
        # }
        payload = json.loads(body)

        # 发送POST请求
        response = requests.post(
            url=url,
            json=payload,  # 自动将字典转换为JSON并设置正确的Content-Type
            headers=headers
        )
        # 检查请求是否成功
        response.raise_for_status()
        # 返回JSON响应
        result = response.json()
    elif active_project_env == "dev":
        result=fake_execute_domain_service_by_entity_id(domain,service,body)

    return result

@tool
def SmartThingsPlannerTool(task) -> str:
    """
    Used to generate a plan of api calls to make to execute a command. The input to this tool is the user command in natural language. Always share the original user command to this tool to provide the overall context.
    """
    llm= get_llm()
    # TODO
    one_liners_string = Device_info_doc.get_one_liners_string()
    device_capability_string = Device_info_doc.get_device_capability_string()
    prompt = f"""You are a planner that helps users interact with their smart devices.
    You are given a list of high level summaries of entity capabilities ("all capabilities:").
    You are also given a list of available entities ("entities you can use") which will tell you the name and entity_id of the entity, as well as listing which capabilities the entity has.
    Your job is to figure out the sequence of which entities and capabilities to use in order to execute the user's command.

    Follow these instructions:
    - Include entity_ids (guid strings), capability ids, and explanations of what needs to be done in your plan.
    - The capability information you receive is not detailed. Often there will be multiple capabilities that sound like they might work. You should list all of the ones that might work to be safe.
    - Don't always assume the devices are already on.

    all capabilities:
    {one_liners_string}

    devices you can use:
    {device_capability_string}

    Use the following format:
    Device Ids: list of relevant devices IDs and names
    Capabilities: list of relevant capabilities
    Plan: steps to execute the command
    Explanation: Any further explanations and notes
    <FINISHED>
    """
    system_message = {
        "role": "system",
        "content": prompt,
    }
    user_message = {
        "role": "user",
        "content": task,
    }
    response = llm.invoke([system_message] +[user_message])
    return response.content

class SmartThingsAgent(BaseToolAgent):
    def get_tools(self) -> List[Callable]:
        tools = [
                SmartThingsPlannerTool,
                 get_all_entity_id,
                 get_services_by_domain,
                 get_states_by_entity_id,
                 execute_domain_service_by_entity_id]
        return tools

    def call_tools(self, state: MessagesState):
        llm = get_llm().bind_tools(self.get_tools())
        prompt = f"""
You are an agent that assists with queries against some API.

Instructions:
- Include a description of what you've done in the final answer, include device IDs
- If you encounter an error, try to think what is the best action to solve the error instead of trial and error.

Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create
Final Answer: the final output from executing the plan. Add <FINISHED> after your final answer.

Your must always output a thought, action, and action input.
Do not forget to say I'm finished when the user's command is executed.
Begin!
"""
        system_message = {
            "role": "system",
            "content": prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        return {"messages": [response]}

@tool
def SmartThingsTool(task) -> str:
    """
    Use this to interact with smartthings. Accepts natural language commands. Do not omit any details from the original command. Use this tool to determine which device can accomplish the query.
    """
    return SmartThingsAgent().run_agent(task)


if __name__ == "__main__":
    print(SmartThingsAgent().run_agent("台灯现在可用吗？"))
    pass