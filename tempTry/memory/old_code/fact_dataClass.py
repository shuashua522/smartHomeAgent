from typing import List
from dataclasses import dataclass, asdict

# ===================== 核心数据结构定义 =====================

@dataclass
class DeviceInfo:
    """设备信息结构化存储"""
    device_id: str  # 设备唯一标识（自动生成/提取）
    environment: List[str] = None  # 设备环境（位置、场景等）
    usage_habits: List[str] = None  # 使用习惯（时间、频率、偏好等）
    capabilities: List[str] = None  # 设备能力（功能、操作等）
    others: List[str] = None        # 其他相关信息

    def __post_init__(self):
        self.environment = self.environment or []
        self.usage_habits = self.usage_habits or []
        self.capabilities = self.capabilities or []
        self.others = self.others or []

@dataclass
class UserProfile:
    """用户画像结构化存储"""
    preferences: List[str] = None    # 用户偏好（如温度偏好、亮度偏好）
    living_habits: List[str] = None  # 生活习惯（如作息、场景使用习惯）
    device_preferences: List[str] = None  # 设备使用偏好
    others: List[str] = None         # 其他用户相关信息

    def __post_init__(self):
        self.preferences = self.preferences or []
        self.living_habits = self.living_habits or []
        self.device_preferences = self.device_preferences or []
        self.others = self.others or []