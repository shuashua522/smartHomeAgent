from typing import List, Callable

from langgraph.graph import MessagesState

from agent_project.agentcore.commons.base_agent import BaseToolAgent
from agent_project.agentcore.commons.utils import get_llm
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.persistent import \
    ConditionCheckerTool, NotifyOnConditionTool
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.smartThings import \
    get_all_entity_id,get_services_by_domain,get_states_by_entity_id,execute_domain_service_by_entity_id


class SingleAgent(BaseToolAgent):
    def __init__(self,logger=None):
        # 第一步：调用父类的__init__，初始化self.logger
        super().__init__(logger=logger)  # 关键：传递logger参数给父类

    def get_tools(self) -> List[Callable]:
        tools = [get_all_entity_id,
                 get_services_by_domain,
                 get_states_by_entity_id,
                 execute_domain_service_by_entity_id,
                 ConditionCheckerTool,
                 NotifyOnConditionTool
                 ]
        return tools

    def call_tools(self, state: MessagesState):
        system_prompt = """
                    你是一款控制智能家居的人工智能。通过调用给定的工具以满足用户需求
                    - 需要持久化时请使用ConditionCheckerTool和NotifyOnConditionTool工具
                """
        llm = get_llm().bind_tools(self.get_tools())
        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        print(response.content)
        return {"messages": [response]}

if __name__ == "__main__":
    SingleAgent().run_agent("网络状态")