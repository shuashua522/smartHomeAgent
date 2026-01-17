from typing import List, Callable

from langgraph.graph import MessagesState

from agent_project.agentcore.commons.base_agent import BaseToolAgent
from agent_project.agentcore.commons.utils import get_llm
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.memory_with_Profile import \
    UserProfileTool
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.persistent import \
    ConditionCheckerTool, NotifyOnConditionTool
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.smartThings import \
    SmartThingsTool


class SageAgent(BaseToolAgent):
    def get_tools(self) -> List[Callable]:
        tools=[
            SmartThingsTool,
            UserProfileTool,
            ConditionCheckerTool,
            NotifyOnConditionTool
        ]
        return tools

    def call_tools(self, state: MessagesState):
        llm = get_llm().bind_tools(self.get_tools())
        prompt = f"""
You are an agent who controls smart homes. You always try to perform actions on their smart devices in response to user input.

Instructions:
- Try to personalize your actions when necessary.
- Plan several steps ahead in your thoughts
- The user's commands are not always clear, sometimes you will need to apply critical thinking
- Tools work best when you give them as much information as possible
- Only perform the task requested by the user, don't schedule additional tasks
- You cannot interact with the user and ask questions.
- You can assume that all the devices are smart.

涉及持久化的操作请直接使用ConditionCheckerTool和NotifyOnConditionTool
不要无端臆想没有提供的设备能力
不要无谓地重复调用工具执行已完成的任务或者执行不了的任务，请合理地根据工具执行结果再决定是否需要继续调用或就此结束
不要试图自己规划设备调用任务
"""
        # You must always output a thought, action, and action input.
        # At last,You must output a Final Answer.
        system_message = {
            "role": "system",
            "content": prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        return {"messages": [response]}

if __name__ == "__main__":
    SageAgent().run_agent("每当网关可用，打开插座")