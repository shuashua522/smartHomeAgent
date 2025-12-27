import logging
import json
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from smartHome.m_agent.common.get_llm import get_llm
from pydantic import BaseModel, Field


from pydantic import BaseModel, Field
from typing import Optional, List

from smartHome.m_agent.common.global_config import GLOBALCONFIG


class EntityFact(BaseModel):
    """
    Home Assistant（HA）实体（Entity）事实性信息模型
    完全对齐HA Entity核心属性，涵盖实体标识、状态、能力、匹配线索等维度：
    1. 实体唯一标识：HA标准Entity ID（domain.object_id），是定位实体的核心
    2. 实体基础属性：所属域、友好名称、设备类型等HA原生属性
    3. 状态相关：可采集的状态类型、当前状态值（贴合HA state/attributes逻辑）
    4. 操作能力：HA实体支持的服务（如light.turn_on、climate.set_temperature）
    5. 实体匹配线索：用于定位Entity ID的多维度信息（友好名、所属空间、用户口语描述等）
    6. 使用习惯：用户操作该HA实体的行为偏好
    7. 补充信息：未归类的HA实体相关属性（如计量单位、联动规则等）
    """
    # 1. HA实体核心标识（替代原device_id）
    entity_id: Optional[str] = Field(
        default=None,
        description="Home Assistant实体唯一标识，格式为domain.id（如binary_sensor.isa_cn_blt_3_1md0u6qht0k00_dw2hl_contact_state_p_2_2）",
        examples=["binary_sensor.isa_cn_blt_3_1md0u6qht0k00_dw2hl_contact_state_p_2_2"]
    )

    friendly_name: Optional[str] = Field(
        default=None,
        description="HA实体的友好名称（用户可见名称，对应HA entity_attributes中的friendly_name）",
        examples=["客厅吸顶灯", "卧室空调", "厨房智能插座"]
    )

    # 3. 状态相关（拆分HA的state和attributes逻辑）
    states: List[str] = Field(
        default=[],
        description="该HA实体可采集的状态类型，如开关状态、亮度、温度等",
        examples=[["开关状态", "音量"]]
    )

    # 4. HA实体操作能力（替代原capabilities，贴合HA Service逻辑）
    capabilities: List[str] = Field(
        default=[],
        description="""该实体实际可执行的功能（中文描述），推导逻辑：
        1. 解析实体attributes中的supported_features十进制数值（如21565），该数值是多个功能特性值的累加结果；
        2. 匹配该实体所属domain（如media_player）的services中，每个服务要求的supported_features特性值；
        3. 筛选出实体supported_features包含的特性值对应的服务，转换为用户易懂的中文功能描述（如“调节音量”对应volume_set服务，需特性值4）。""",
        examples=[
            ["调节音量", "播放前一首", "播放/暂停音乐", "设置静音"],
            ["打开音箱", "关闭音箱", "切下一首"]
        ]
    )

    # 5. 实体匹配线索（替代原device_id_clues，贴合HA场景）
    entity_matching_clues: List[str] = Field(
        default=[],
        description="用于定位HA Entity ID的多维度线索，涵盖友好名、所属空间、domain、用户口语描述等",
        examples=[
            ["客厅吸顶灯", "living_room", "light", "暖光"],
            ["卧室空调", "bedroom", "climate", "变频"]
        ]
    )

    others: List[str] = Field(
        default=[],
        description="未归类的HA实体相关事实性信息，如联动规则、设备型号、所属设备集成等",
        examples=[
            ["设备型号：Yeelight YLXD01YL", "HA集成：yeelight", "联动规则：开门触发light.turn_on"],
            ["HA集成：miot", "供电方式：插座供电", "品牌：格力"]
        ]
    )

    # 同步更新示例为HA标准格式
    # 重新生成的model_config：完全匹配你修改后的字段
    model_config = {
        "json_schema_extra": {
            "examples": [
                # 示例1：传感器类实体（匹配你entity_id的示例格式）
                {
                    "entity_id": "binary_sensor.isa_cn_blt_3_1md0u6qht0k00_dw2hl_contact_state_p_2_2",
                    "friendly_name": "门窗传感器",
                    "states": ["开关状态", "电量"],
                    "capabilities": [],  # 传感器类实体无主动操作能力，留空
                    "entity_matching_clues": ["门窗传感器", "living_room", "binary_sensor", "接触状态"],
                    "others": ["HA集成：isa", "设备型号：ISA-CN-BLT-3", "供电方式：纽扣电池"]
                },
                # 示例2：灯光类实体（匹配你capabilities/states的示例风格）
                {
                    "entity_id": None,  # 模拟未获取到entity_id的场景
                    "friendly_name": "客厅吸顶灯",
                    "states": ["开关状态", "音量"],  # 匹配你states字段的示例
                    "capabilities": ["light.turn_on", "light.turn_off", "light.set_brightness", "light.set_color_temp"],
                    "entity_matching_clues": ["客厅吸顶灯", "living_room", "light", "暖光"],
                    "others": ["设备型号：Yeelight YLXD01YL", "HA集成：yeelight", "联动规则：开门触发light.turn_on"]
                },
                # 示例3：空调类实体（补充多场景参考）
                {
                    "entity_id": None,
                    "friendly_name": "卧室空调",
                    "states": ["开关状态", "温度", "风速"],
                    "capabilities": ["climate.set_temperature", "climate.set_fan_mode", "climate.turn_on"],
                    "entity_matching_clues": ["卧室空调", "bedroom", "climate", "变频"],
                    "others": ["HA集成：miot", "供电方式：插座供电", "品牌：格力"]
                }
            ]
        }
    }


logger=GLOBALCONFIG.logger

# 2. 模拟你的业务逻辑（替换为你实际的result获取逻辑）
# 假设result["structured_response"]是EntityFact实例
# 这里仅做示例，你需保留自己的result获取代码
result = {
    "structured_response": EntityFact(
        entity_id="binary_sensor.isa_cn_blt_3_1md0u6qht0k00_dw2hl_contact_state_p_2_2",
        friendly_name="门窗传感器",
        states=["开关状态", "电量"],
        capabilities=[],
        entity_matching_clues=["门窗传感器", "living_room", "binary_sensor", "接触状态"],
        others=["HA集成：isa", "设备型号：ISA-CN-BLT-3", "供电方式：纽扣电池"]
    )
}
entity_fact = result["structured_response"]

# 3. 日志输出entity_fact（三种常用方式，按需选择）
def log_entity_fact(entity_fact: EntityFact, logger: logging.Logger):
    """
    输出EntityFact实例到日志，提供不同粒度的输出方式
    """
    # 方式1：DEBUG级别 - 输出格式化的JSON字符串（最详细，易读，适合调试）
    # model_dump_json：Pydantic内置方法，将模型转为格式化的JSON字符串
    entity_json = entity_fact.model_dump_json(indent=2, ensure_ascii=False)
    logger.debug("解析得到HA EntityFact完整信息：\n%s", entity_json)

    # 方式2：INFO级别 - 输出核心字段（简洁，适合日常日志）
    core_info = {
        "entity_id": entity_fact.entity_id,
        "friendly_name": entity_fact.friendly_name,
        "main_state": entity_fact.states[0] if entity_fact.states else None,
        "capabilities_count": len(entity_fact.capabilities)
    }
    logger.info("HA实体核心信息：%s", core_info)

    # 方式3：WARNING/ERROR级别 - 关键字段单独输出（聚焦重点，适合异常场景）
    if not entity_fact.entity_id:
        logger.warning("HA实体缺少核心标识！friendly_name=%s, matching_clues=%s",
                       entity_fact.friendly_name, entity_fact.entity_matching_clues)

# 调用函数输出日志
log_entity_fact(entity_fact, logger)