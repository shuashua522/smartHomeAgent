import logging
import os
import sys
from io import StringIO
import subprocess
import configparser
import os
from langchain.chat_models import init_chat_model
from typing import Literal, List, Callable, Dict
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_project.agentcore.commons.base_agent import BaseToolAgent
from agent_project.agentcore.commons.utils import get_llm, get_null_logger
import agent_project.agentcore.config.global_config as global_config

import re
from pathlib import Path
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.persistent import \
    ConditionCheckerTool, NotifyOnConditionTool
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.smartThings import \
    get_all_entity_id,get_services_by_domain,get_states_by_entity_id,execute_domain_service_by_entity_id


def get_logger():
    logger=global_config.GLOBAL_AGENT_DETAILED_LOGGER
    if logger==None:
        logger=get_null_logger()
    return logger
class ClarificationResponse(BaseModel):
    response: str = Field(
        description="Answer by YES or NO if the task is possible or the query can be answered"
    )
    explanation: str = Field(
        description="Return an explanation of the response to the user"
    )

class FilteringResponse(BaseModel):
    devices: list[str] = Field(
        description="A list that contains the devices that can potentially accomplish the task"
    )


class PrePlanningResponse(BaseModel):
    response: str = Field(
        description="Respond, IN ONE WORD, whether the instruction is associated with sensor (the retrieval of information from a sensor), control (the control of a device), or persistent (complex goals that demand the creation of automation routines)."
    )

class PlanningResponse(BaseModel):
    output: str = Field(
        description=""
    )

class ReadingResponse(BaseModel):
    output: str = Field(
        description=""
    )
class PersistentResponse(BaseModel):
    output: str = Field(
        description=""
    )


class clarification(BaseToolAgent):

    def get_tools(self) -> List[Callable]:
        tools = [get_all_entity_id,
                 get_services_by_domain,
                 get_states_by_entity_id]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions="""
        # 请严格按照以下格式要求返回结果:
        # - 仅返回YES 或 NO。判断任务是否可行或查询是否可回答，用 YES 或 NO 作答。
        # - 无论发生任何情况，你最终的答案只能是YES 或 NO"""

        format_instructions=PydanticOutputParser(
            pydantic_object=ClarificationResponse
        ).get_format_instructions()

        system_prompt = f"""
        你是一款控制智能家居的人工智能。你可以通过调用给定的工具来获取设备信息。
        请回答：当前设备是否能成功执行用户的指令。

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

class filtering(BaseToolAgent):

    def get_tools(self) -> List[Callable]:
        tools = [get_all_entity_id,
                 get_services_by_domain,
                 get_states_by_entity_id]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions="请按照以下格式要求返回结果:包含有潜力完成该任务的设备的列表"
        format_instructions=PydanticOutputParser(
            pydantic_object=FilteringResponse
        ).get_format_instructions()
        system_prompt = f"""
        你是一个控制智能家居的人工智能（AI）。 你可以通过调用给定的工具来获取设备信息。
        需提取出与执行用户指令相关的设备名称。  
      
        请通过返回部分设备（即相关设备子集）来回应。

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

class pre_planning(BaseToolAgent):
    def __init__(self, devices):
        self.devices=devices

    def get_tools(self) -> List[Callable]:
        tools = [get_states_by_entity_id]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions = """
        # 请按照以下格式要求返回结果:
        # - 用一个词回应(sensor | control | persistent，三者中的一个)，说明该指令与sensor（从传感器获取信息）、control（控制设备）还是persistent（需要创建自动化流程的复杂目标）相关。
        # - """
        format_instructions = PydanticOutputParser(
            pydantic_object=PrePlanningResponse
        ).get_format_instructions()
        system_prompt = f"""
        你是一款控制智能家居的人工智能（AI）。你将收到一组设备清单（清单上的设备即可完成用户指令）。

        设备清单：{self.devices}

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

class planning(BaseToolAgent):
    def __init__(self, devices):
        self.devices=devices

    def get_tools(self) -> List[Callable]:
        tools = [get_services_by_domain,
                 get_states_by_entity_id,
                 execute_domain_service_by_entity_id]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions = """"""
        format_instructions = PydanticOutputParser(
            pydantic_object=PlanningResponse
        ).get_format_instructions()
        system_prompt = f"""
        你是一款控制智能家居的人工智能（AI）。你将收到一组设备清单（清单上的设备即可完成用户指令）。
        你的任务是通过调用工具使用这些设备以完成指令。  

        筛选后的设备：{self.devices}  

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

class reading(BaseToolAgent):
    def __init__(self, devices):
        self.devices=devices

    def get_tools(self) -> List[Callable]:
        tools = [get_services_by_domain,
                 get_states_by_entity_id,
                 ]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions = """"""
        format_instructions = PydanticOutputParser(
            pydantic_object=ReadingResponse
        ).get_format_instructions()
        system_prompt = f"""
        你是一款控制智能家居的人工智能（AI）。你将收到一组设备清单（清单上的设备即可完成用户指令）。
        你的任务是通过调用工具从设备状态中获取相关信息以完成任务。

        筛选后的设备：{self.devices}  

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

class persistent(BaseToolAgent):
    def __init__(self, devices):
        self.devices=devices

    def get_tools(self) -> List[Callable]:
        tools = [
                ConditionCheckerTool,
                NotifyOnConditionTool
                 ]
        return tools

    def call_tools(self, state: MessagesState):
        # format_instructions = """"""
        format_instructions = PydanticOutputParser(
            pydantic_object=PersistentResponse
        ).get_format_instructions()
        system_prompt = f"""
        你是一款控制智能家居的人工智能（AI）。你将收到一组设备清单（清单上的设备即可完成用户指令）。
        你接收用户指令，并以此为依据创建自动化流程。
        
        对设备与传感器进行分析，提出一个传感器触发条件，以及基于该触发条件应如何调整设备（状态/设置）。

        筛选后的设备：{self.devices}  

        {format_instructions}
        """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

def run_sashaAgent(command:str):
    get_logger().info("=========clarification 阶段==========")
    clarification_answer=clarification().run_agent(command)
    clarification_answer=PydanticOutputParser(pydantic_object=ClarificationResponse).parse(clarification_answer)

    if clarification_answer.response == "YES":
        # filtering step
        get_logger().info("=========filtering 阶段==========")
        filtering_answer = filtering().run_agent(command)
        filtering_answer = PydanticOutputParser(pydantic_object=FilteringResponse).parse(filtering_answer)
        devices=filtering_answer.devices

        # pre_planning
        get_logger().info("=========pre_planning 阶段==========")
        pre_planning_answer = pre_planning(devices=devices).run_agent(command)
        pre_planning_answer = PydanticOutputParser(pydantic_object=PrePlanningResponse).parse(pre_planning_answer)

        assert pre_planning_answer.response in [
            "control",
            "sensor",
            "persistent",
        ], "Pre-Planning response failed and didn't return Control or Sensor"

        if pre_planning_answer.response == "control":
            # planning step
            get_logger().info("=========planning 阶段==========")
            planning_answer = planning(devices).run_agent(command)
            planning_answer = PydanticOutputParser(pydantic_object=PlanningResponse).parse(planning_answer)

        elif pre_planning_answer.response == "sensor":
            # sensor reading prompt
            get_logger().info("=========reading 阶段==========")
            planning_answer = reading(devices).run_agent(command)
            planning_answer = PydanticOutputParser(pydantic_object=ReadingResponse).parse(planning_answer)

        elif pre_planning_answer.response == "persistent":
            get_logger().info("=========persistent 阶段==========")
            planning_answer = persistent(devices).run_agent(command)
            planning_answer = PydanticOutputParser(pydantic_object=PersistentResponse).parse(planning_answer)
    else:
        return {"output": clarification_answer.explanation}

if __name__ == "__main__":
    run_sashaAgent("每当网关可用，打开插座")