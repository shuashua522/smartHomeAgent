import os
import re
import traceback

from smartHome.m_agent.agent.home_agent import run_ourAgent
from smartHome.m_agent.common.global_config import GLOBALCONFIG
from smartHome.m_agent.common.logger import setup_dynamic_indent_logger
from smartHome.m_agent.test.baselines_homeassitant.sage.sage_coordinator import run_sageAgent
from smartHome.m_agent.test.baselines_homeassitant.sashaAgent import run_sashaAgent


def process_testcases(agent_name,testNums=1):
    """
    遍历测试用例
    agent_name取值：[singleAgent,sashaAgent,ourAgent]
    """
    if agent_name not in ["singleAgent","sashaAgent","sageAgent","ourAgent"]:
        raise ValueError("无效的arg：agent_name")

    # 遍历测试用例
    from test_cases import init_devices
    for index, func in enumerate(init_devices.registered_functions):
        # if(index+1<=36):
        #     continue
        if(index+1<testNums):
            continue
        if(index+1>testNums):
            continue
        question=func()

        # 处理文件名：移除特殊字符，确保文件名合法 ; 保留中文、字母、数字和下划线，其他字符替换为下划线
        cleaned_name = re.sub(r'[^\w\u4e00-\u9fa5]', '', question)
        # 生成文件名：序号_清洗后的内容（如"0_将整个房子变暗"、"1_网络状况"）
        filename = f"{index+1}_{cleaned_name}.log"

        # 日志配置
        GLOBALCONFIG.nested_logger=setup_dynamic_indent_logger(logger_name="agent_init_dialogue", log_file_path=f"logs/{GLOBALCONFIG.model}/{agent_name}/{filename}")

        # todo 真正测试前开启
        os.environ["LANGSMITH_PROJECT"] = agent_name

        if(index+1>=26 and index+1<=34):
            question="持久化："+question
        try:
            if agent_name=="singleAgent":
                # SingleAgent(logger=logger).run_agent(question)
                pass
            elif agent_name=="sashaAgent":
                run_sashaAgent(question)
            elif agent_name=="sageAgent":
                run_sageAgent(question)
            elif agent_name=="ourAgent":
                run_ourAgent(question)
        except Exception as e:
            # 1. 获取完整的异常信息（类型、消息、堆栈跟踪）
            # traceback.format_exc() 会返回包含堆栈的字符串，便于调试
            exception_detail = traceback.format_exc()

            # 2. 将异常原样打印到日志（使用error级别，突出异常）
            GLOBALCONFIG.print_nested_log(
                f"执行Agent [{agent_name}] 时发生异常：\n"
                f"异常类型：{type(e).__name__}\n"
                f"异常消息：{str(e)}\n"
                f"完整堆栈跟踪：\n{exception_detail}"
            )

def main(agent_name,testNums):
    process_testcases(agent_name=agent_name,testNums=testNums)


if __name__=="__main__":
    # main("singleAgent",1)
    # main("sashaAgent",1)
    # main("sageAgent",1)

    main("ourAgent",46)
    pass