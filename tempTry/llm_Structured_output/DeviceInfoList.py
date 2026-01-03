from typing import List
from pydantic import BaseModel, Field

# 先复用之前定义的两个模型（省略重复注释，保持完整性）
class DeviceInfo(BaseModel):
    """单个智能家居设备的完整信息模型（包含ID、名称、选择理由）"""
    device_id: str = Field(
        description="设备唯一标识ID",
        examples=["31ae92d8a163d77f8d6a5741c0d1b89c"]
    )
    device_name: str = Field(
        description="设备名称",
        examples=["客厅智能吸顶灯"]
    )
    device_reason: str = Field(
        description="选择该设备的理由（50字以内）",
        examples=["亮度可调节，能匹配客厅日常照明和观影场景需求"]
    )

class DeviceIdList(BaseModel):
    """多个智能家居设备的事实性信息列表模型"""
    devices: List[DeviceInfo] = Field(
        default=[],
        description="所有候选设备的完整信息列表（ID、名称、选择理由）",
        examples=[
            [
                {
                    "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",
                    "device_name": "客厅智能吸顶灯",
                    "device_reason": "亮度可调节，能匹配客厅日常照明和观影场景需求"
                }
            ]
        ]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"devices": [{"device_id": "31ae92d8a163d77f8d6a5741c0d1b89c", "device_name": "客厅智能吸顶灯", "device_reason": "亮度可调节，能匹配客厅日常照明和观影场景需求"}]}]
        }
    }

# 1. 创建DeviceIdList实例（构造测试数据）
device_list_instance = DeviceIdList(
    devices=[
        DeviceInfo(
            device_id="31ae92d8a163d77f8d6a5741c0d1b89c",
            device_name="客厅智能吸顶灯",
            device_reason="亮度可调节，能匹配客厅日常照明和观影场景需求"
        ),
        DeviceInfo(
            device_id="31ae92d8a163d77f8d6a54856d1b89c",
            device_name="卧室智能窗帘",
            device_reason="支持定时开合，能配合作息自动调节卧室采光"
        )
    ]
)

# 2. 转为JSON字符串（核心方法：model_dump_json()）
# 无缩进（紧凑格式，适合传输/存储）
json_str_compact = device_list_instance.model_dump_json()
# 带缩进（美化格式，适合调试/查看）
json_str_pretty = device_list_instance.model_dump_json(indent=4)

# 3. 输出结果
print("=== 紧凑格式JSON字符串 ===")
print(json_str_compact)
print("\n=== 美化格式JSON字符串 ===")
print(json_str_pretty)