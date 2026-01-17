import traceback
from typing import List, Callable, Any

from langgraph.graph import MessagesState

from agent_project.agentcore.commons.base_agent import BaseToolAgent
from agent_project.agentcore.commons.utils import get_llm
from langchain_core.tools import tool

from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.homeAssitant_api_func import \
    run_code
from agent_project.agentcore.smart_home_agent.test_with_baselines.baselines_homeassitant.sage.smart.smartThings import \
    SmartThingsPlannerTool, get_all_entity_id, get_services_by_domain, get_states_by_entity_id, \
    execute_domain_service_by_entity_id

@tool
def PythonInterpreterTool(command):
    """
    Use this tool to run python code to automate tasks when necessary.
    Use like python_interpreter(<python code string here>). Make sure the last line is the entrypoint to your code.
    """
    command = command.replace("```", "").replace("python", "")
    try:
        # 将代码分割为“定义部分”和“执行部分”：按最后一个换行符拆分
        # 假设最后一行为执行入口（如函数调用），其余为定义（如函数、变量定义）
        code_define, code_run = command.strip("\n").rsplit("\n", 1)

        # 处理打印语句：若执行部分是 print(...)，则提取括号内的内容作为实际执行代码
        # 例如将 print(my_function()) 转换为 my_function()
        if code_run[:5] == "print":
            code_run = code_run[6:-1]

        # 提取执行部分的函数名：从执行代码（如 my_function(123)）中解析出函数名（my_function）
        # 用于后续将代码信息注册到 code_registry 中
        fn_name = (
            code_run.split("(")[0]  # 按左括号分割，取左侧部分（函数名+可能的空白符）
            .replace("\n", "")  # 移除换行符
            .replace(" ", "")  # 移除空格
            .replace("\t", "")  # 移除制表符
        )

        # 执行代码：先执行定义部分（如函数定义），再执行执行部分（如函数调用），获取结果
        result = run_code(code_define, code_run)

        # 返回代码执行结果
        return result
    except Exception:
        # 若执行过程中出错，返回错误堆栈信息（便于调试）
        return traceback.format_exc()

class ConditionCheckerAgent(BaseToolAgent):
    def get_tools(self) -> List[Callable]:
        tools=[
            SmartThingsPlannerTool,
            get_all_entity_id,
            get_services_by_domain,
            get_states_by_entity_id,
            execute_domain_service_by_entity_id,
            PythonInterpreterTool,]
        return tools

    def call_tools(self, state: MessagesState):
        llm = get_llm().bind_tools(self.get_tools())
        prompt = f"""
You are an agent that writes code that checks whether a condition is satisfied by querying some API.
I have encapsulated the API calls into functions and also added them to your tool list. You can directly call these functions—get_all_entity_id, get_services_by_domain, get_states_by_entity_id, and execute_domain_service_by_entity_id—in your code.
You should first plan the sequence of API calls you need to make. Then you should get detailed
documentation for each of the API calls. Finally, you should write python code to check the condition. Remember, the code:
- should be tested using a python interpreter
- should involve the checking of a single condition that involves checking the state of a single device that best fulfills the user request
- should follow the correct dictionary key structure
- the function must return True if the device state condition is satisfied and False otherwise
- the code should only include a single function definition that does not include any arguments.
- the last line of the programe should be an example of the function call

If the code doesn't work, you should try to figure out why and fix it. Your final answer should include the current status of the condition and the name of the function you wrote to check it. Remember, the function should return True or False.

Starting below, you should follow this format:

User query: the query a User wants help with related to the API

Thought: you should always think about what to do

Action: the action to take, should be one of the tools

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create

Final Answer: the final output from executing the plan. This must always include the name of the function
you wrote to check the condition.

Your must always output a thought, action, and action input.

Begin!
"""
        system_message = {
            "role": "system",
            "content": prompt,
        }
        response = llm.invoke([system_message] + state["messages"])
        return {"messages": [response]}

@tool
def ConditionCheckerTool(text):
    """
    Use this tool to check whether a certain condition is satisfied. Accepts natural language commands. Returns the name of a function which checks the condition. Inputs should be phrased as questions. In addition to the question that needs to be checked, you should also provide any extra information that might contextualize the question.
    """
    question=f"""
    Condition to check: {text}
    Thought: I should generate a plan to help with checking this condition.
    """
    return ConditionCheckerAgent().run_agent(question)

@tool
def NotifyOnConditionTool(function_name,notify_when,condition_description,action_description):
    """
    Use this tool to get notified when a condition occurs. It should be called keys:
    function_name (str): name of the function that checks the condition
    notify_when (bool): [true, false]
    condition_description (str)
    action_description (str): what to do when the condition is met
    """
    return f"You will be notified when the condition occurs."