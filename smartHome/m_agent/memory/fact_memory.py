import os
import sys
import uuid
from typing import List, Optional

from langchain.agents import create_agent
from pydantic import BaseModel, Field, ValidationError
import json

from smartHome.m_agent.agent.langchain_middleware import log_before, AgentContext, log_response, log_before_agent, \
    log_after_agent
from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.common.global_config import GLOBALCONFIG
from smartHome.m_agent.common.logger import setup_dynamic_indent_logger
from smartHome.m_agent.memory.device_info import DEVICEINFO
from smartHome.m_agent.memory.vector_device import VECTORDB, TextWithMeta, search_topK_device_by_clues, add, delete, \
    update
from langchain.tools import tool

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

    device_name: str = Field(
        default=None,
        description="智能家居设备的名称（如电视机）",
        examples=["电视机"]
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
                    "device_name": "多功能灯",
                    "states": ["亮度", "色温", "温度"],  # 匹配states字段的示例风格
                    "capabilities": ["调节色温", "调节亮度", "开关控制客厅加湿器"],  # 复用capabilities字段示例
                    "device_id_clues": ["客厅灯", "living_room", "智能吸顶灯", "暖光"],  # 复用clues字段示例
                    "usage_habits": ["每天22:00关闭", "睡前调至暖光模式", "周末全天开启"],  # 复用习惯字段示例
                    "others": ["设备型号：MI-Light-01", "供电方式：插座供电"]  # 复用others字段示例
                },
                # 补充第二个示例（覆盖另一类设备，提升参考性）
                {
                    "device_id": "87bf45e9c2d78a10b3e56f89c7a2d4e8",
                    "device_name": "万用空调",
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

class SmartHomeMemory():
    def __init__(self):
        # key是设备ID，value是列表，所包含的实体的fact
        self.entities_fact={}
        # key是设备ID，value是设备的fact
        self.device_fact={}

        self.entities_fact_save_path="./temp_output/entities_fact.json"
        self.device_fact_save_path="./temp_output/device_fact.json"
        self.vector_db=VECTORDB

    def init_memory_for_device(self):
        llm = get_llm()

        with open(self.entities_fact_save_path, "r", encoding="utf-8") as f:
            # 解析JSON内容到字典
            init_entities_fact = json.load(f)
        # 存储最终的设备事实信息（key=device_id，value=设备级事实性信息）
        device_fact_dict = {}

        # 2. 遍历每个设备，拼接提示词并调用LLM提取设备事实
        for device_id, entity_fact_list in init_entities_fact.items():
            # 2.1 拼接该设备下所有实体的事实信息（转为易读的文本）
            device_name=DEVICEINFO.get_device_detail(device_id)["name"]
            # entity_info_text = self._format_entity_fact_list(entity_fact_list)
            if (GLOBALCONFIG.env == "test"):
                GLOBALCONFIG.nested_logger = GLOBALCONFIG.memory_init_logger
                # 设计LLM提示词模板（聚焦设备级事实提取）
                prompt = f"""
        请基于以下智能家居设备（device_id: {device_id} ({device_name})）包含的所有实体事实性信息，提取该设备的**整体事实性信息**:
    
        【设备包含的实体事实信息】
        {entity_fact_list}
                    """.strip()
                prompt = f"""
                请基于以下智能家居设备（device_id: {device_id} ({device_name})）包含的所有实体事实性信息，分析该设备的**整体事实性信息**:
                1. device_id_clues里只需包含设备名字
                2. usage_habits应该为空，因为实体信息里不可能包含
                3. 该实体实际可执行的功能，没有则为空，如调节亮度，调节温度等
                4. 只提取实体包含的事实信息，不要过多分析、假设
                最终输出严格符合JSON格式（需包含功能分析结果，字段与DeviceFact模型对齐）
                - 字段内容与DeviceFact模型的描述和示例一致

                【设备包含的实体事实信息】
                {entity_fact_list}
                """
                agent = create_agent(
                    model=get_llm(),
                    response_format=DeviceFact,  # 多实体列表格式
                    middleware=[log_before, log_response, log_before_agent, log_after_agent],
                    context_schema=AgentContext
                )

                result = agent.invoke(
                    input={"messages": [
                        {"role": "system", "content": prompt},
                    ]},
                    context=AgentContext(agent_name="设备事实_记忆初始化阶段")
                )


                device_fact = result["structured_response"]
            else:
                device_fact = DeviceFact(
                    device_id=device_id,
                    device_name=device_name,
                    states=["tryi"],
                    capabilities=[],
                    device_id_clues=[],
                    usage_habits=[],
                    others=[]
                )
            device_fact_dict[device_id]=device_fact

        self.device_fact = device_fact_dict
        self._save_init_device_fact_to_json(device_fact_dict, self.device_fact_save_path)
        self._save_init_device_fact_to_vector_db()


    def init_memory_for_entity(self):
        """
        依据设备-实体包含映射表，提取出设备所包含的实体的所有事实性信息
        :return:
        """
        llm=get_llm()

        init_fact={}
        for device_id in DEVICEINFO.device_entity_mapping:
            device_name=DEVICEINFO.get_device_detail(device_id)["name"]
            device_entities_fact=[]
            for entity_id in DEVICEINFO.device_entity_mapping[device_id]:
                entity_detail=DEVICEINFO.get_entity_detail(entity_id)
                domain_service=DEVICEINFO.get_domain_service(entity_id)
                if(GLOBALCONFIG.env=="test"):
                    GLOBALCONFIG.nested_logger = GLOBALCONFIG.memory_init_logger
                    agent = create_agent(
                        model=get_llm(),
                        response_format=EntityFact,  # 多实体列表格式
                        middleware=[log_before, log_response, log_before_agent, log_after_agent],
                        context_schema=AgentContext
                    )
                    prompt = f"""
                    解析下面这个homeassitant实体，分析：
                    1. 该HA实体可采集的状态类型，如开关状态、亮度、温度等
                    2. 该实体实际可执行的功能，没有则为空，如调节亮度，调节温度等
                    3. 用于定位HA Entity ID的多维度线索，没有则为空。对于homeassitant实体，只需包含其friendly_name即可
                    4. 只提取实体包含的事实信息，不要过多分析、假设
                    最终输出严格符合JSON格式（需包含功能分析结果，字段与EntityFact模型对齐）
                    - 字段内容与EntityFact模型的描述和示例一致
                    - 字段内容要简练，不需要像这样额外描述，"用户口语示例：'门窗光照','门窗传感器 光照度','窗户光线 强/弱"，应该为"门窗光照"、'门窗传感器 光照度'
                    - 字段内容不需要包含当前设备的具体状态数值，比如"当前状态：弱","更新时间:2025-12-1"，这些具体数值都不应该包含。

                    【entity】
                    {entity_detail}
                    【service】
                    {domain_service}
                    """
                    result = agent.invoke(
                        input={"messages": [
                            {"role": "system", "content": prompt},
                        ]},
                        context=AgentContext(agent_name="实体事实_记忆初始化阶段")
                    )

                    # 解析并输出提取结果（适配EntityFact字段）
                    entity_fact = result["structured_response"]
                else:
                    entity_fact=EntityFact(
                        entity_id=entity_detail["entity_id"],
                        friendly_name=entity_detail["attributes"]["friendly_name"],
                        states=[entity_detail["state"]],
                        capabilities=[],
                        entity_matching_clues=[],
                        others=[]
                    )
                device_entities_fact.append(entity_fact)

            init_fact[device_id]=device_entities_fact

        self._save_init_entities_fact_to_json(
            init_fact=init_fact,
            save_path=self.entities_fact_save_path  # 目标保存路径
        )
        self.entities_fact=init_fact

    def _save_init_device_fact_to_vector_db(self):
        """
        修正版：将DeviceFact实例的点语法访问替代字典下标访问，解决TypeError
        """
        # 步骤1：定义「字段名」与「对应布尔标识」的映射表（保持不变）
        field_boolean_mapping = [
            ("states", {"states": True}),
            ("capabilities", {"capabilities": True}),
            ("device_id_clues", {"device_id_clues": True}),
            ("usage_habits", {"usage_habits": True}),
            ("others", {"others": True})
        ]

        # 步骤2：遍历所有设备Fact（value是DeviceFact实例）
        for device_id, device_fact in self.device_fact.items():
            # 跳过无效设备ID或非DeviceFact实例
            if not device_id or not isinstance(device_fact, DeviceFact):
                print(f"⚠️  无效设备ID「{device_id}」或非DeviceFact实例，跳过入库")
                continue

            # 步骤3：获取/创建设备专属集合（点语法访问DeviceFact属性）
            device_name = device_fact.device_name or "N/A"  # 替代 device_fact["device_name"]
            collection = self.vector_db.get_or_create_collection(device_id, device_name)

            # 步骤4：遍历映射表，统一处理所有字段（点语法访问列表属性）
            for field_name, boolean_kwargs in field_boolean_mapping:
                # 用getattr获取DeviceFact的列表属性（替代 device_fact[field_name]）
                content_list = getattr(device_fact, field_name, [])
                if not isinstance(content_list, list) or not content_list:
                    continue

                # 步骤5：遍历内容列表，创建TextWithMeta并入库（保持不变）
                for content in content_list:
                    # 生成唯一text_id
                    text_id = uuid.uuid4().hex

                    # 实例化TextWithMeta（强转content为字符串，避免报错）
                    text_with_meta = TextWithMeta(
                        text_id=text_id,
                        content=str(content),
                        **boolean_kwargs
                    )

                    # 入库
                    self.vector_db.add_text_to_vector_db(text_with_meta, collection)

    def _save_init_device_fact_to_json(self, init_fact: dict, save_path: str):
        try:
            # 步骤1：确保目标目录存在（不存在则创建）
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"自动创建目录：{save_dir}")

            # 步骤2：将init_fact中的DeviceFact实例转为字典（Pydantic序列化）
            init_fact_serializable = {}
            for device_id, device_fact in init_fact.items():
                # 对每个EntityFact实例调用model_dump()转为字典
                init_fact_serializable[device_id] = device_fact.model_dump()

            # 步骤3：写入JSON文件（处理中文+格式化）
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    init_fact_serializable,
                    f,
                    ensure_ascii=False,  # 保证中文正常显示（不转义为\uXXX）
                    indent=2,  # 格式化输出，易读
                    sort_keys=False  # 保持键的原有顺序（比如device_id的遍历顺序）
                )

            print(f"init_fact已成功保存到：{os.path.abspath(save_path)}")

        except Exception as e:
            print(f"保存init_fact到JSON失败：{e}", file=sys.stderr)
            raise  # 可选：抛出异常让上层处理，根据业务需求调整
    def _save_init_entities_fact_to_json(self, init_fact: dict, save_path: str):
        """
        将包含EntityFact实例的init_fact字典保存为JSON文件

        参数：
        - init_fact: 待保存的字典（key=device_id, value=EntityFact实例列表）
        - save_path: 保存路径（如./temp_output/xx.json）
        """
        try:
            # 步骤1：确保目标目录存在（不存在则创建）
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"自动创建目录：{save_dir}")

            # 步骤2：将init_fact中的EntityFact实例转为字典（Pydantic序列化）
            # 遍历每个device_id对应的EntityFact列表，逐个转字典
            init_fact_serializable = {}
            for device_id, entity_fact_list in init_fact.items():
                # 对每个EntityFact实例调用model_dump()转为字典
                init_fact_serializable[device_id] = [
                    fact.model_dump() for fact in entity_fact_list
                ]

            # 步骤3：写入JSON文件（处理中文+格式化）
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    init_fact_serializable,
                    f,
                    ensure_ascii=False,  # 保证中文正常显示（不转义为\uXXX）
                    indent=2,  # 格式化输出，易读
                    sort_keys=False  # 保持键的原有顺序（比如device_id的遍历顺序）
                )

            print(f"init_fact已成功保存到：{os.path.abspath(save_path)}")

        except Exception as e:
            print(f"保存init_fact到JSON失败：{e}", file=sys.stderr)
            raise  # 可选：抛出异常让上层处理，根据业务需求调整

    def extract_and_update(self,dialogue_record:str):
        """
        通过当前对话，和最近几条对话，提取事实性信息，
        并将事实性信息更新到对应的设备ID中，
        :param dialogue_record:
        :return:
        """

        # 包含多个设备的对话记录（客厅灯+卧室空调+厨房插座）
        # dialogue_record = """
        # 用户：我家客厅的智能灯能调亮度和色温，每天晚上10点我都会把它关掉，睡前还会调成暖光模式，这个灯我叫客厅灯。
        # 客服：您还有其他智能家居设备的使用习惯吗？
        # 用户：卧室的空调我每天早上7点打开，调26℃，能调风速和定时；厨房的插座是智能的，能远程开关，我一般叫它厨房智能插座。
        # """
        prompt=f"""     
        根据当前对话和近期的历史对话，分析是否有新增/过时/修改的事实信息，如果有：
        1. 对话里没有提供设备ID时，调用工具获取到该设备ID
        2. 选择并调用add/delete/update工具对记忆库中的信息进行更新
        3. 更新成功后简单说明本次更新了哪些内容。
        【对话】
        {dialogue_record}
        """
        GLOBALCONFIG.nested_logger=GLOBALCONFIG.memory_update_logger
        agent = create_agent(model=get_llm(),
                             tools=[search_topK_device_by_clues, add, delete,update],
                             middleware=[log_before, log_response, log_before_agent, log_after_agent],
                             context_schema=AgentContext
                             )
        result = agent.invoke(
            input={"messages": [
                {"role": "system", "content": prompt},
            ]},
            context=AgentContext(agent_name="对话__记忆更新阶段")
        )

        # device_fact_list_result = result["structured_response"]
        return result["messages"][-1].content



SMARTHOMEMEMORY=SmartHomeMemory()

@tool
def get_device_all_entities_states(device_id: str):
    """
    获取:设备ID下的所有实体各自可以获取到的状态
    :param device_id: 设备ID
    :return:
    """
    # collection = VECTORDB.client.get_collection(
    #     name=device_id,
    #     embedding_function=VECTORDB.embedding_func
    # )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "temp_output","entities_fact.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load() 将文件对象解析为Python字典/列表
        data = json.load(f)
    ans_str_list = []
    for entities in data[device_id]:
        e_str=f"{entities['entity_id']}({entities['friendly_name']}):{'、'.join(entities['states'])}"
        ans_str_list.append(e_str)
    return "\n".join(ans_str_list)

@tool
def get_device_all_entities_capabilities(device_id: str):
    """
    获取:设备ID下的所有实体各自的能力
    :param device_id: 设备ID
    :return:
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "temp_output", "entities_fact.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load() 将文件对象解析为Python字典/列表
        data = json.load(f)
    ans_str_list = []
    for entities in data[device_id]:
        e_str = f"{entities['entity_id']}({entities['friendly_name']}):{'、'.join(entities['capabilities'])}"
        ans_str_list.append(e_str)
    return "\n".join(ans_str_list)

def load_json_and_convert_dialogues():
    """
        加载JSON文件，将每段对话转换为一段格式化字符串
        :param json_file_path: JSON文件的路径（相对路径或绝对路径）
        :return: 包含每段对话字符串的列表，若执行失败返回空列表
        """
    json_file_path="./dialogue_records.json"
    # 初始化返回结果列表
    dialogue_str_list = []

    try:
        # 1. 打开并加载JSON文件（with语句自动关闭文件，更安全）
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dialogue_json = json.load(f)

        # 2. 遍历JSON中的每一段对话（外层数组的每个子数组）
        for single_dialogue in dialogue_json:
            # 初始化单段对话的拼接字符串
            dialogue_content = ""

            # 3. 遍历单段对话中的每个角色消息（user/ai）
            for msg_obj in single_dialogue:
                # 提取user或ai的内容，避免KeyError
                if "user" in msg_obj:
                    dialogue_content += f"用户：{msg_obj['user']}\n"
                elif "ai" in msg_obj:
                    dialogue_content += f"AI：{msg_obj['ai']}\n"

            # 4. 去除末尾多余的换行符，添加到结果列表（可选，提升整洁度）
            dialogue_str_list.append(dialogue_content.rstrip('\n'))

        return dialogue_str_list

    except FileNotFoundError:
        print(f"错误：未找到指定的JSON文件 -> {json_file_path}")
        return []
    except json.JSONDecodeError:
        print("错误：JSON文件格式无效，无法解析")
        return []
    except Exception as e:
        print(f"错误：执行过程中出现未知异常 -> {str(e)}")
        return []

if __name__ == "__main__":
    # SMARTHOMEMEMORY.init_memory_for_entity()
    # SMARTHOMEMEMORY.init_memory_for_device()

    dialogue_str_list=load_json_and_convert_dialogues()
    for idx, dialogue_str in enumerate(dialogue_str_list, start=1):
        if idx < 7:
            continue
        SMARTHOMEMEMORY.extract_and_update(dialogue_record=dialogue_str)
    pass