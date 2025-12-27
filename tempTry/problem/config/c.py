# 导入a.py中的config实例
from a import config

# 定义一个函数，用于修改config的属性
def modify_config(new_name):
    print("修改前 - c.py 中的config:", config)
    # 修改config实例的属性
    config.app_name = new_name
    print("修改后 - c.py 中的config:", config)