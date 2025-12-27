class Config:
    def __init__(self):
        # 初始化一个属性
        self.app_name = "default_app"

    def __repr__(self):
        # 方便打印查看属性
        return f"Config(app_name='{self.app_name}')"


# 创建Config类的实例
config = Config()