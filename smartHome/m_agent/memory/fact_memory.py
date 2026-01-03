import os
import sys
import uuid
from typing import List, Optional

from langchain.agents import create_agent
from pydantic import BaseModel, Field, ValidationError
import json

from smartHome.m_agent.common.get_llm import get_llm
from smartHome.m_agent.common.global_config import GLOBALCONFIG
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
                # 设计LLM提示词模板（聚焦设备级事实提取）
                prompt = f"""
        请基于以下智能家居设备（device_id: {device_id} ({device_name})）包含的所有实体事实性信息，提取该设备的**整体事实性信息**:
    
        【设备包含的实体事实信息】
        {entity_fact_list}
                    """.strip()
                agent = create_agent(
                    model=get_llm(),
                    response_format=DeviceFact,  # 多实体列表格式
                )
                result = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
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
        self._save_init_device_fact_to_vector_db()
        self._save_init_device_fact_to_json(device_fact_dict,self.device_fact_save_path)

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
                    agent = create_agent(
                        model=get_llm(),
                        response_format=EntityFact,  # 多实体列表格式
                    )
                    prompt = f"""
                    请严格按照以下步骤分析并提取Home Assistant实体的事实性信息：

                    ### 步骤1：分析【entity】内容
                    仔细解析【entity】中的字段（如entity_id、attributes、state等），明确该实体的基础属性（如设备类型、状态维度、所属域）。

                    ### 步骤2：分析【service】内容
                    解析【service】中该实体所属domain支持的所有服务，区分“该实体实际支持的服务”和“该实体不支持的服务”（基于entity的supported_features、attributes等判断）。

                    ### 步骤3：总结功能范围
                    基于前两步的分析，清晰总结：
                    1. 该实体**支持的功能**（需对应到具体的用户可操作行为，如“打开/关闭灯光”“调节温度”）；
                    2. 该实体**不支持的功能**（需明确排除的行为，如“该传感器仅支持读取状态，不支持任何主动操作”“空调不支持调节湿度”）。

                    ### 步骤4：输出结构化信息
                    最终输出严格符合以下JSON格式（需包含功能分析结果，字段与EntityFact模型对齐）：

                    【entity】
                    {entity_detail}
                    【service】
                    {domain_service}

                    ### 强制要求
                    1. 必须先完成功能分析，再输出JSON；
                    2. JSON字段完整，无缺失（空值用空列表/空字符串）；
                    3. 仅输出最终的JSON内容，无任何额外解释、分析文字；
                    4. “支持/不支持的功能”需准确对应【service】和【entity】的匹配结果，不虚构。
                    """
                    result = agent.invoke({
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
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

    def extract(self,dialogue_record:str):
        """
        通过当前对话，和最近几条对话，提取事实性信息，
        并将事实性信息绑定到对应的设备ID中，
        :param dialogue_record:
        :return:
        """
        agent = create_agent(
            model=get_llm(),
            response_format=DeviceFactList,  # 替换为DeviceFactList
        )

        # 包含多个设备的对话记录（客厅灯+卧室空调+厨房插座）
        # dialogue_record = """
        # 用户：我家客厅的智能灯能调亮度和色温，每天晚上10点我都会把它关掉，睡前还会调成暖光模式，这个灯我叫客厅灯。
        # 客服：您还有其他智能家居设备的使用习惯吗？
        # 用户：卧室的空调我每天早上7点打开，调26℃，能调风速和定时；厨房的插座是智能的，能远程开关，我一般叫它厨房智能插座。
        # """
        prompt="""
        根据当前对话和近期的历史对话，分析是否有新增/过时/修改的事实信息，如果有：
        1. 调用工具获取到该设备ID
        2. 调用add/delete/update工具对记忆库中的信息进行处理
        """

        agent = create_agent(model=get_llm(),
                             tools=[search_topK_device_by_clues, add, delete,update], )
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": "广州天气怎么样"},
            ]
        })

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
        return device_fact_list_result

    def update(self,device_fact_list:str):
        """

        :param new_memory:
        :return:
        """

        for new_device_fact in device_fact_list:
            # 核心判断：device_id存在且不是空字符串
            if new_device_fact.get("device_id") and new_device_fact["device_id"].strip():
                # todo 使用llm依据new_device_fact调用 _retrieve_topk_similar_devices方法检索出最相似的topk设备候选，并从中挑出最佳的一个设备
                new_device_fact["device_id"]=None

        for new_device_fact in device_fact_list:
            device_id=new_device_fact["device_id"]
            old_device_fact=self.device_fact["device_id"]
            # todo 让llm根据new_device_fact更新old_device_fact,输出格式：DeviceFact
            updated_device_fact=None
            self.device_fact["device_id"]=updated_device_fact



SMARTHOMEMEMORY=SmartHomeMemory()

@tool
def get_device_all_entities_states(device_id: str):
    """
    获取:设备ID下的所有实体各自可以获取到的状态
    :param device_id: 设备ID
    :return:
    """
    pass

@tool
def get_device_all_entities_capabilities(device_id: str):
    """
    获取:设备ID下的所有实体各自的能力
    :param device_id: 设备ID
    :return:
    """
    pass

if __name__ == "__main__":
    # SmartHomeMemory().init_memory_by_deviceInfo()
    SmartHomeMemory().init_memory_for_entity()
    SmartHomeMemory().init_memory_for_device()
    VECTORDB.print_all_collections_content()
    pass