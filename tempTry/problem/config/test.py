from b import show_config
from c import modify_config

# 第一步：查看b.py中初始的config
print("=== 初始状态 ===")
show_config()

# 第二步：用c.py修改config属性
print("\n=== 执行c.py的修改操作 ===")
modify_config("my_new_app")

# 第三步：再次查看b.py中的config，验证是否感知到修改
print("\n=== 修改后查看b.py的config ===")
show_config()