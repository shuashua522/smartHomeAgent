import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
import os

from tempTry.memory.get_llm import get_llm

# 加载环境变量（Mem0推荐的配置方式）
load_dotenv()


# ===================== 1. 结构化模型定义（Mem0核心：Pydantic验证） =====================
class DeviceFact(BaseModel):
    """设备事实结构化模型（Mem0风格：强类型+字段说明）"""
    device_id: str = Field(description="设备唯一标识（如客厅灯、卧室空调，自动归一化别名）")
    environment: List[str] = Field(default=[], description="设备环境：位置/场景/联动关系（如「客厅灯安装在客厅天花板」）")
    usage_habits: List[str] = Field(default=[], description="使用习惯：时间/频率/操作偏好（如「每天睡前关客厅灯」）")
    capabilities: List[str] = Field(default=[], description="设备能力：可执行的功能（如「卧室空调可调节温度/风速」）")
    others: List[str] = Field(default=[], description="其他设备相关信息")

class UserProfileFact(BaseModel):
    """用户画像结构化模型"""
    preferences: List[str] = Field(default=[], description="用户偏好：温度/亮度/模式等（如「喜欢空调调26度」）")
    living_habits: List[str] = Field(default=[], description="生活习惯：作息/场景使用（如「每天7点起床」）")
    device_preferences: List[str] = Field(default=[], description="设备偏好：喜欢/讨厌的设备（如「偏爱新热水器」）")
    others: List[str] = Field(default=[], description="其他用户相关信息")

class SmartHomeMemory(BaseModel):
    """智能家居记忆总模型（Mem0核心记忆结构）"""
    devices: Dict[str, DeviceFact] = Field(default={}, description="设备事实字典（key=device_id）")
    user_profile: UserProfileFact = Field(default_factory=UserProfileFact, description="用户画像")
    update_time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             description="最后更新时间")
    dialogue_history: List[str] = Field(default=[], description="关联的对话历史（Mem0要求保留溯源信息）")

    # Mem0风格：增量合并新记忆
    def merge(self, new_memory: "SmartHomeMemory") -> None:
        """
        增量合并新提取的记忆（核心逻辑：去重+合并+冲突标记）
        参考Mem0的memory.merge()实现
        """
        # 1. 合并设备事实
        for dev_id, new_dev_fact in new_memory.devices.items():
            if dev_id not in self.devices:
                self.devices[dev_id] = new_dev_fact
            else:
                # 去重合并列表（Mem0核心：避免重复记忆）
                self.devices[dev_id].environment = list(
                    set(self.devices[dev_id].environment + new_dev_fact.environment))
                self.devices[dev_id].usage_habits = list(
                    set(self.devices[dev_id].usage_habits + new_dev_fact.usage_habits))
                self.devices[dev_id].capabilities = list(
                    set(self.devices[dev_id].capabilities + new_dev_fact.capabilities))
                self.devices[dev_id].others = list(set(self.devices[dev_id].others + new_dev_fact.others))

        # 2. 合并用户画像
        self.user_profile.preferences = list(set(self.user_profile.preferences + new_memory.user_profile.preferences))
        self.user_profile.living_habits = list(
            set(self.user_profile.living_habits + new_memory.user_profile.living_habits))
        self.user_profile.device_preferences = list(
            set(self.user_profile.device_preferences + new_memory.user_profile.device_preferences))
        self.user_profile.others = list(set(self.user_profile.others + new_memory.user_profile.others))

        # 3. 合并对话历史（Mem0要求保留溯源）
        self.dialogue_history = list(set(self.dialogue_history + new_memory.dialogue_history))

        # 4. 更新时间
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===================== 2. Mem0风格Prompt模板（核心：引导LLM输出结构化内容） =====================
# 完全参考Mem0的Prompt设计：清晰指令+输入示例+输出Schema+格式约束
SMART_HOME_EXTRACTION_PROMPT = """
### 任务说明
你是智能家居事实提取专家，需要从用户对话中提取结构化的设备事实和用户画像，严格遵循以下规则：
1. 设备ID自动归一化别名（如「客厅的灯」→「客厅灯」、「主卧空调」→「卧室空调」）；
2. 提取的信息必须来自对话文本，不编造、不推测；
3. 输出仅保留JSON，无任何多余文字（如解释、备注）；
4. 列表内容去重，每个条目简洁（10-20字）。

### 输出Schema（严格遵循）
{{
  "devices": {{
    "设备ID": {{
      "device_id": "归一化后的设备ID",
      "environment": ["环境信息1", "环境信息2"],
      "usage_habits": ["使用习惯1", "使用习惯2"],
      "capabilities": ["能力1", "能力2"],
      "others": ["其他信息1"]
    }}
  }},
  "user_profile": {{
    "preferences": ["用户偏好1"],
    "living_habits": ["生活习惯1"],
    "device_preferences": ["设备偏好1"],
    "others": []
  }},
  "dialogue_history": ["输入的对话文本"],
  "update_time": "{current_time}"
}}

### 输入示例
用户对话：我每天睡前关客厅的灯，客厅灯装在天花板，卧室空调能调26度，我喜欢26度的空调温度。

### 输出示例
{{
  "devices": {{
    "客厅灯": {{
      "device_id": "客厅灯",
      "environment": ["客厅灯安装在客厅天花板"],
      "usage_habits": ["每天睡前关闭客厅灯"],
      "capabilities": [],
      "others": []
    }},
    "卧室空调": {{
      "device_id": "卧室空调",
      "environment": [],
      "usage_habits": [],
      "capabilities": ["可调节温度至26度"],
      "others": []
    }}
  }},
  "user_profile": {{
    "preferences": ["喜欢空调温度调至26度"],
    "living_habits": [],
    "device_preferences": [],
    "others": []
  }},
  "dialogue_history": ["我每天睡前关客厅的灯，客厅灯装在天花板，卧室空调能调26度，我喜欢26度的空调温度。"],
  "update_time": "2025-12-15 10:00:00"
}}

### 待处理对话
{dialogue_text}
"""

# ===================== 3. LLM提取器（Mem0核心：LLM调用+记忆解析） =====================
class Mem0StyleSmartHomeExtractor:
    """参考Mem0的MemoryExtractor实现：LLM驱动的智能家居记忆提取"""

    def __init__(self, llm_provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """
        初始化提取器（支持OpenAI/本地Ollama）
        :param llm_provider: 可选"openai" / "ollama"
        :param model: 模型名称（openai: gpt-3.5-turbo/gpt-4; ollama: llama3:8b/phi3:mini）
        """


    def _clean_llm_output(self, output: str) -> str:
        """Mem0风格：清理LLM输出（移除多余文字，只保留JSON）"""
        # 提取JSON块（处理LLM可能输出的markdown包裹）
        json_match = re.search(r"\{[\s\S]*\}", output)
        if json_match:
            return json_match.group(0)
        return output

    def _call_llm(self, dialogue_text: str) -> str:
        """调用LLM（兼容OpenAI/Ollama）"""
        prompt = SMART_HOME_EXTRACTION_PROMPT.format(
            dialogue_text=dialogue_text,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        messages = [
            {"role": "system", "content": "你是严格遵循指令的结构化数据提取助手"},
            {"role": "user", "content": prompt}
        ]
        llm=get_llm()
        response = llm.invoke(messages).content
        print(response)
        return response

    def extract(self, dialogue_text: str) -> SmartHomeMemory:
        """
        核心提取方法（Mem0风格：LLM提取→结构化验证→返回记忆对象）
        :param dialogue_text: 用户对话文本
        :return: 结构化的智能家居记忆
        """
        # 1. 调用LLM
        raw_output = self._call_llm(dialogue_text)
        # 2. 清理输出
        clean_output = self._clean_llm_output(raw_output)

        try:
            # 3. 解析为字典
            memory_dict = json.loads(clean_output)
            # 4. Pydantic验证（Mem0核心：强类型校验）
            # 先处理devices字典（DeviceFact列表）
            validated_devices = {}
            for dev_id, dev_data in memory_dict.get("devices", {}).items():
                validated_devices[dev_id] = DeviceFact(**dev_data)
            # 处理user_profile
            validated_user_profile = UserProfileFact(**memory_dict.get("user_profile", {}))
            # 构建最终记忆对象
            memory = SmartHomeMemory(
                devices=validated_devices,
                user_profile=validated_user_profile,
                dialogue_history=memory_dict.get("dialogue_history", [dialogue_text]),
                update_time=memory_dict.get("update_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            return memory

        except (json.JSONDecodeError, ValidationError) as e:
            # Mem0风格：异常处理+降级返回空记忆
            print(f"LLM输出解析失败：{e}，返回空记忆")
            return SmartHomeMemory(dialogue_history=[dialogue_text])

    def to_json(self, memory: SmartHomeMemory, indent: int = 4) -> str:
        """Mem0风格：记忆对象序列化为JSON"""
        return memory.model_dump_json(indent=indent, ensure_ascii=False)

    def from_json(self, json_str: str) -> SmartHomeMemory:
        """Mem0风格：从JSON恢复记忆对象"""
        memory_dict = json.loads(json_str)
        return SmartHomeMemory(**memory_dict)


# ===================== 4. 使用示例（完全复刻Mem0的使用流程） =====================
if __name__ == "__main__":
    # -------------------- 示例1：使用OpenAI GPT-3.5/4 --------------------
    # 需提前设置环境变量：OPENAI_API_KEY（或在.env文件中配置）
    extractor = Mem0StyleSmartHomeExtractor()

    # 模拟多轮对话（Mem0支持增量记忆）
    dialogue_1 = """
    我每天晚上睡前都会关掉客厅的灯，客厅灯安装在客厅天花板中间。
    卧室空调可以调节温度和风速，我喜欢把卧室空调调到26度。
    扫地机器人能自动清洁地板，我习惯每周五让它打扫一次。
    """
    dialogue_2 = """
    客厅灯还能和窗帘联动，天黑的时候自动打开。
    我不喜欢用厨房的旧热水器，更喜欢新的那款。
    扫地机器人还支持定时清扫功能，我通常设置在下午3点。
    """

    # 第一轮提取
    memory_v1 = extractor.extract(dialogue_1)
    print("=== 第一轮提取结果 ===")
    print(extractor.to_json(memory_v1))

    # 第二轮增量更新（Mem0核心：合并新记忆）
    memory_v2 = extractor.extract(dialogue_2)
    memory_v1.merge(memory_v2)  # 增量合并
    print("\n=== 增量更新后结果 ===")
    print(extractor.to_json(memory_v1))

    # 持久化到文件（Mem0推荐的记忆存储方式）
    with open("smart_home_memory.json", "w", encoding="utf-8") as f:
        f.write(extractor.to_json(memory_v1))

    # 从文件加载记忆
    with open("smart_home_memory.json", "r", encoding="utf-8") as f:
        loaded_memory = extractor.from_json(f.read())
    print("\n=== 从文件加载的记忆 ===")
    print(extractor.to_json(loaded_memory))

    # -------------------- 示例2：使用本地Ollama（Llama3） --------------------
    # 需先启动ollama：ollama run llama3:8b
    # extractor_ollama = Mem0StyleSmartHomeExtractor(llm_provider="ollama", model="llama3:8b")
    # memory_ollama = extractor_ollama.extract(dialogue_1)
    # print("\n=== 本地LLM提取结果 ===")
    # print(extractor_ollama.to_json(memory_ollama))