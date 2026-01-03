from pydantic import BaseModel, Field
from langchain.agents import create_agent
from smartHome.m_agent.common.get_llm import get_llm
from pydantic import BaseModel, Field


class DeviceFact(BaseModel):
    """
    智能家居设备事实性信息模型
    整合设备核心属性与用户关联信息，包含五大核心维度：
    1. 设备唯一标识：用于精准定位设备
    2. 设备状态能力：可采集/上报的状态信息
    3. 设备操作能力：自身可执行的功能及联动能力
    4. 设备定位线索：用于匹配设备ID的多维度特征
    5. 设备使用习惯：用户使用该设备的行为偏好
    6. 其他信息：未归类的设备相关事实性内容
    """
    # 设备唯一标识
    device_id: str = Field(
        default=None,
        description="智能家居设备唯一标识（如Home Assistant设备ID：31ae92d8a163d77f8d6a5741c0d1b89c）",
        examples=["31ae92d8a163d77f8d6a5741c0d1b89c"]
    )

    # 设备可获取的状态信息
    states: list[str] = Field(
        default=[],
        description="该设备可采集/上报的事实性状态信息（如开关状态、亮度、温度等）",
        examples=[["亮度", "色温", "温度"], ["音量", "湿度50%"]]
    )

    # 设备可执行的功能（含联动）
    capabilities: list[str] = Field(
        default=[],
        description="设备自身可执行的功能，或通过该设备触发的联动操作（如调节自身参数、控制关联设备）",
        examples=[["调节色温", "调节亮度", "开关控制客厅加湿器"], ["定时开关", "联动门窗传感器触发报警"]]
    )

    # 设备ID定位线索
    device_id_clues: list[str] = Field(
        default=[],
        description="用于定位/匹配设备ID的多维度线索，涵盖设备名称、所属空间、类型、用户口语描述等",
        examples=[["客厅灯", "living_room", "智能吸顶灯", "暖光"], ["卧室空调", "bedroom", "变频空调"]]
    )

    # 用户使用习惯
    usage_habits: list[str] = Field(
        default=[],
        description="用户使用该设备的行为习惯，包含时间、频率、操作偏好等事实性信息",
        examples=[["每天22:00关闭", "睡前调至暖光模式", "周末全天开启"], ["每周一三五18:00自动开启"]]
    )

    # 其他设备相关信息
    others: list[str] = Field(
        default=[],
        description="未归类的设备相关事实性信息，补充上述维度未覆盖的内容",
        examples=[["设备型号：MI-Light-01", "供电方式：插座供电"]]
    )

    # 重新生成的model_config：完全匹配修改后的字段示例
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",  # 匹配你指定的HA设备ID格式
                    "states": ["亮度", "色温", "温度"],  # 匹配states字段的示例风格
                    "capabilities": ["调节色温", "调节亮度", "开关控制客厅加湿器"],  # 复用capabilities字段示例
                    "device_id_clues": ["客厅灯", "living_room", "智能吸顶灯", "暖光"],  # 复用clues字段示例
                    "usage_habits": ["每天22:00关闭", "睡前调至暖光模式", "周末全天开启"],  # 复用习惯字段示例
                    "others": ["设备型号：MI-Light-01", "供电方式：插座供电"]  # 复用others字段示例
                },
                # 补充第二个示例（覆盖另一类设备，提升参考性）
                {
                    "device_id": "87bf45e9c2d78a10b3e56f89c7a2d4e8",
                    "states": ["音量", "湿度50%"],
                    "capabilities": ["定时开关", "联动门窗传感器触发报警"],
                    "device_id_clues": ["卧室空调", "bedroom", "变频空调"],
                    "usage_habits": ["每周一三五18:00自动开启"],
                    "others": ["设备品牌：格力", "联网方式：WiFi"]
                }
            ]
        }
    }

# 新增：定义多设备的列表模型（核心修改）
class DeviceFactList(BaseModel):
    """多个智能家居设备的事实性信息列表模型"""
    device_facts: list[DeviceFact] = Field(
        default=[],
        description="包含对话中所有设备的事实性信息，每个设备对应一个DeviceFact实例",
        examples=[
            # 示例：2个设备的列表
            [
                {
                    "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",
                    "states": ["亮度", "色温"],
                    "capabilities": ["调节亮度", "调节色温"],
                    "device_id_clues": ["客厅灯", "living_room"],
                    "usage_habits": ["每天22:00关闭"],
                    "others": []
                },
                {
                    "device_id": None,
                    "states": ["温度", "风速"],
                    "capabilities": ["调节温度", "定时开关"],
                    "device_id_clues": ["卧室空调", "bedroom"],
                    "usage_habits": ["每天7:00开启"],
                    "others": ["品牌：美的"]
                }
            ]
        ]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "device_facts": [
                        {
                            "device_id": "31ae92d8a163d77f8d6a5741c0d1b89c",
                            "states": ["亮度", "色温"],
                            "capabilities": ["调节亮度", "调节色温"],
                            "device_id_clues": ["客厅灯", "living_room"],
                            "usage_habits": ["每天22:00关闭"],
                            "others": []
                        },
                        {
                            "device_id": None,
                            "states": ["温度", "风速"],
                            "capabilities": ["调节温度", "定时开关"],
                            "device_id_clues": ["卧室空调", "bedroom"],
                            "usage_habits": ["每天7:00开启"],
                            "others": ["品牌：美的"]
                        }
                    ]
                }
            ]
        }
    }

# 创建Agent：响应格式改为多设备列表模型（核心修改）
agent = create_agent(
    model=get_llm(),
    response_format=DeviceFactList,  # 替换为DeviceFactList
)

# 包含多个设备的对话记录（客厅灯+卧室空调+厨房插座）
dialogue_record_01 = """
用户：我家客厅的智能灯能调亮度和色温，每天晚上10点我都会把它关掉，睡前还会调成暖光模式，这个灯我叫客厅灯。
客服：您还有其他智能家居设备的使用习惯吗？
用户：卧室的空调我每天早上7点打开，调26℃，能调风速和定时；厨房的插座是智能的，能远程开关，我一般叫它厨房智能插座。
"""
dialogue_record_02 = """
用户：打开床边的灯。
客服：灯有两个，我不清楚哪个是床边的灯。灯列表：灯泡、台灯
用户：台灯
用户：插座上连着冰箱
用户: 2号插座不要关闭
"""
dialogue_record=dialogue_record_02

# 优化提取指令：明确要求识别多个设备并分别提取
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""从以下对话记录中提取**所有智能家居设备**的事实性信息，严格遵守：
【核心规则】
1. 识别对话中提及的每一个独立设备，为每个设备生成一个独立的DeviceFact实例；
2. 每个DeviceFact仅填充该设备对应的信息，未提及的字段保留默认值（device_id为None，列表字段为空）；
3. 字段内容简化为简短字符串，符合各字段的描述要求（如使用习惯仅保留核心行为）；
4. 不主观添加对话中未提及的信息，仅提取事实性内容。

【对话记录】
{dialogue_record}"""
        }
    ]
})

# 解析并输出多个设备的提取结果（核心修改）
device_fact_list_result = result["structured_response"]
print("=== 提取的多个智能家居设备事实性信息 ===")
devices_fact_str=device_fact_list_result.model_dump_json(indent=4)
print(devices_fact_str)
# 遍历每个设备的DeviceFact
for idx, device_fact in enumerate(device_fact_list_result.device_facts, 1):
    print(f"\n【设备{idx}】")
    print(f"  设备唯一标识：{device_fact.device_id}")
    print(f"  可采集状态：{device_fact.states}")
    print(f"  可执行功能：{device_fact.capabilities}")
    print(f"  定位线索：{device_fact.device_id_clues}")
    print(f"  使用习惯：{device_fact.usage_habits}")
    print(f"  其他信息：{device_fact.others}")