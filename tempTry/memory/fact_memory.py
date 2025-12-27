from typing import List
from pydantic import BaseModel, Field, ValidationError
import json

from tempTry.memory.device_info import DEVICEINFO
from tempTry.memory.get_llm import get_llm


class DeviceFact(BaseModel):
    """
    其实就4种加一个用户画像，
    设备能力、设备获取状态、定位设备的信息、设备使用习惯
    """
    device_id: str = Field(description="设备唯一标识")
    environment: List[str] = Field(default=[], description="设备环境：位置/场景/联动关系（如「客厅灯安装在客厅天花板」）")
    states: List[str] = Field(default=[], description="设备环境：位置/场景/联动关系（如「客厅灯安装在客厅天花板」）")
    usage_habits: List[str] = Field(default=[], description="使用习惯：时间/频率/操作偏好（如「每天睡前关客厅灯」）")
    capabilities: List[str] = Field(default=[], description="设备能力：可执行的功能（如「卧室空调可调节温度/风速」）或者可以影响的设备")
    product_manual: List[str] = Field(default=[], description="产品说明书信息：存储说明书提供的所有相关内容（如规格参数、安装要求、维保说明、功能介绍等）")
    others: List[str] = Field(default=[], description="其他设备相关信息")
class SmartHomeMemory():
    def init_memory_by_deviceInfo(self):
        """
        从设备表、实体表、和api得到的entity、service，得到设备能力信息和设备名字，
        如果设备注册表里有产品说明书，从产品说明书上得到补充性信息
        :return:
        """
        llm=get_llm()

        for device_id in DEVICEINFO.device_entity_mapping:
            device_name=DEVICEINFO.get_device_detail(device_id)["name"]
            device_capability_str=""
            for entity_id in DEVICEINFO.device_entity_mapping[device_id]:
                entity_detail=DEVICEINFO.get_entity_detail(entity_id)
                domain_service=DEVICEINFO.get_domain_service(entity_id)

                # messages = [
                #     {"role": "system", "content": """
                #                 你是HomeAssistant实体能力分析专家，需按以下规则分析：
                #                 1. 用简洁的中文描述实体具备的核心能力（如"开关控制、亮度调节"）
                #                 2. 只输出能力描述，不添加多余解释
                #                 3. 能力描述不超过50字
                #                 """},
                #     {"role": "user", "content": f"entity_detail: {entity_detail}\ndomain_service: {domain_service}"}
                # ]
                # response=llm.invoke(messages)
                # entity_capability=response.content
                # print(entity_id)
                entity_capability=entity_detail["attributes"]["friendly_name"]
                device_capability_str += f"- 实体 {entity_id}：{entity_capability}\n"
            print(f"{device_name}--{device_capability_str}")
        pass

    def select_most_appropriate_device_from_descr_AND_deviceList(self,description: str, devices: List[str]) :
        """从描述信息和设备id列表中选择一个最恰当的设备
        返回设备ID和理由
        """
        # 匹配逻辑（如关键词匹配、相似度计算等）
        pass

    def select_most_appropriate_device_from_descr(self,description: str) :
        """从描述信息中选择一个最恰当的设备
        返回设备ID和理由
        """
        # 匹配逻辑（如关键词匹配、相似度计算等）
        pass

    def select_devices_from_descr(self,description: str) :
        """
        从描述信息中选择符合的设备列表，并说明原因
        :param description:
        :return:
        """
        pass

    def extract(self,message:str):
        """
        通过当前对话，和最近几条对话，提取事实性信息，
        并将事实性信息绑定到对应的设备ID中，
        :param message:
        :return:
        """
        pass
    def update(self,new_memory:str):
        """

        :param new_memory:
        :return:
        """
        pass

if __name__ == "__main__":
    SmartHomeMemory().init_memory_by_deviceInfo()
    pass