import json
import os


class DeviceInfo():
    def __init__(self):
        self.devices=self.load_devices()
        self.entities=self.load_entitied()
        self.device_entity_mapping=self.init_device_entity_mapping()
        self.domains_services=self.load_domains_services()
        self.save_all_instance_vars_to_json()

    def get_device_detail(self,device_id):
        for device in self.devices:
            if device_id==device["id"]:
                return device
    def get_entity_detail(self,entity_id):
        for entity in self.entities:
            if entity["entity_id"]==entity_id:
                return entity

    def get_domain_service(self, entity_id):
        domain = entity_id.split(".")[0]
        for domain_service in self.domains_services:
            if domain_service["domain"]==domain:
                return domain_service
    def _load_from_json(self,file_name):
        # 1. 获取当前py文件的绝对目录路径
        # os.path.abspath(__file__)：获取当前py文件的完整绝对路径
        # os.path.dirname()：提取路径中的目录部分（去掉文件名）
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 拼接目标json文件的完整路径（跨平台兼容）
        # os.path.join()：自动适配不同系统的路径分隔符
        json_file_path = os.path.join(current_script_dir, "copied_data", f"{file_name}.json")

        # 3. 打开文件并加载json数据
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_devices(self):
        device_registry = self._load_from_json("device_registry")
        devices = device_registry["data"]["devices"]
        return devices

    def load_entitied(self):
        entities = self._load_from_json("entities")
        return entities

    def load_domains_services(self):
        domains_services = self._load_from_json("domains_services")
        return domains_services

    def has_entity(self,entity_id):
        for entity in self.entities:
            if entity["entity_id"] == entity_id:
                return True
        return False
    def init_device_entity_mapping(self):
        entity_registry=self._load_from_json("entity_registry")["data"]["entities"]
        device_entity_mapping = {}
        for device in self.devices:
            device_id = device["id"]
            # 存储设备核心信息 + 空的实体列表
            device_entity_mapping[device_id] = []
        # 2. 遍历实体，关联到对应的设备
        for entity in entity_registry:
            entity_device_id = entity.get("device_id")
            # 只处理有设备关联的实体（排除device_id为None的实体）
            if entity_device_id and entity_device_id in device_entity_mapping:
                entity_id=entity.get("entity_id")
                if self.has_entity(entity_id):
                    # 将实体添加到对应设备的实体列表中
                    device_entity_mapping[entity_device_id].append(entity_id)

        return device_entity_mapping

    def save_all_instance_vars_to_json(self):
        """核心函数：将所有实例变量分别保存到JSON文件，文件名=变量名"""
        # 1. 获取所有实例变量（self.__dict__ 存储实例的属性键值对）

        instance_vars = {
            var_name: var_value
            for var_name, var_value in self.__dict__.items()
        }

        # 2. 遍历每个实例变量，逐个保存为JSON文件
        for var_name, var_value in instance_vars.items():
            # 构建文件路径：./temp_output/变量名.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "temp_output", f"{var_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(var_value, f, ensure_ascii=False, indent=2)

DEVICEINFO=DeviceInfo()

if __name__ == "__main__":
    print(DEVICEINFO.get_domain_service("light"))
    for device_id in DEVICEINFO.device_entity_mapping:
        print(device_id)
    print(DEVICEINFO.get_entity_detail("light.yeelink_cn_1162511951_mbulb3_s_2"))
    pass