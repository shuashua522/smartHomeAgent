import configparser
import os

from smartHome.m_agent.common.logger import get_logger


class Global_Config():
    def __init__(self):
        self.configparser=self.load_configparser()
        self.provider = self.configparser.get("base", 'selected_llm_provider')
        self.model = self.configparser.get(self.provider, 'model')
        self.base_url = self.configparser.get(self.provider, 'base_url')
        self.api_key = self.configparser.get(self.provider, 'api_key')
        self.logger = get_logger("my_test","logs/my_test.log")
        self.memory_logger= get_logger("memory_update","logs/memory_update.log")
        self.memory_init_logger= get_logger("memory_init","logs/memory_init.log")
        # env取值：dev，test，prod
        self.env="test"
    def load_configparser(self):
        # 获取当前文件(global_config.py)的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 定位到配置文件所在目录（与global_config.py同目录）
        config_dir = os.path.dirname(current_file_path)
        # 拼接配置文件的绝对路径
        llm_config_file_path = os.path.join(config_dir, 'llm_config.ini')

        # 创建配置解析器对象
        llm_configparser = configparser.ConfigParser()
        # 读取INI文件
        llm_configparser.read(llm_config_file_path, encoding='utf-8')

        # 设置LangSmith跟踪开关和API密钥和标签
        os.environ["LANGSMITH_TRACING"] = llm_configparser.get('LangSmith', 'langsmith_tracing')
        os.environ["LANGSMITH_API_KEY"] = llm_configparser.get('LangSmith', 'langsmith_api_key')

        return llm_configparser

GLOBALCONFIG=Global_Config()