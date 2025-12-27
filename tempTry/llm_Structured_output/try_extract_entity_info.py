from pydantic import BaseModel, Field
from langchain.agents import create_agent
from smartHome.m_agent.common.get_llm import get_llm
from pydantic import BaseModel, Field


from pydantic import BaseModel, Field
from typing import Optional, List


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



# 创建Agent：响应格式改为多设备列表模型（核心修改）
# 3. 创建Agent（适配多实体列表）
agent = create_agent(
    model=get_llm(),
    response_format=EntityFact,  # 多实体列表格式
)

demo_entity="""
{
    "entity_id": "media_player.xiaomi_cn_701074704_l15a",
    "state": "idle",
    "attributes": {
      "volume_level": 0.1,
      "is_volume_muted": true,
      "media_content_type": "music",
      "device_class": "speaker",
      "friendly_name": "小米AI音箱（第二代）  音箱",
      "supported_features": 21565
    },
    "last_changed": "2025-11-04T06:52:46.891045+00:00",
    "last_reported": "2025-11-04T06:53:46.985629+00:00",
    "last_updated": "2025-11-04T06:53:46.985629+00:00",
    "context": {
      "id": "01K96T9939471HASGJX3ZS1HGR",
      "parent_id": null,
      "user_id": null
    }
  }
"""
demo_service="""
{
    "domain": "media_player",
    "services": {
      "turn_on": {
        "name": "Turn on",
        "description": "Turns on the power of the media player.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                128
              ]
            }
          ]
        }
      },
      "turn_off": {
        "name": "Turn off",
        "description": "Turns off the power of the media player.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                256
              ]
            }
          ]
        }
      },
      "toggle": {
        "name": "Toggle",
        "description": "Toggles a media player on/off.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                384
              ]
            }
          ]
        }
      },
      "volume_up": {
        "name": "Turn up volume",
        "description": "Turns up the volume.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                4,
                1024
              ]
            }
          ]
        }
      },
      "volume_down": {
        "name": "Turn down volume",
        "description": "Turns down the volume.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                4,
                1024
              ]
            }
          ]
        }
      },
      "media_play_pause": {
        "name": "Play/Pause",
        "description": "Toggles play/pause.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                16385
              ]
            }
          ]
        }
      },
      "media_play": {
        "name": "Play",
        "description": "Starts playing.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                16384
              ]
            }
          ]
        }
      },
      "media_pause": {
        "name": "Pause",
        "description": "Pauses.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                1
              ]
            }
          ]
        }
      },
      "media_stop": {
        "name": "Stop",
        "description": "Stops playing.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                4096
              ]
            }
          ]
        }
      },
      "media_next_track": {
        "name": "Next",
        "description": "Selects the next track.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                32
              ]
            }
          ]
        }
      },
      "media_previous_track": {
        "name": "Previous",
        "description": "Selects the previous track.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                16
              ]
            }
          ]
        }
      },
      "clear_playlist": {
        "name": "Clear playlist",
        "description": "Removes all items from the playlist.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                8192
              ]
            }
          ]
        }
      },
      "volume_set": {
        "name": "Set volume",
        "description": "Sets the volume level.",
        "fields": {
          "volume_level": {
            "required": true,
            "selector": {
              "number": {
                "min": 0,
                "max": 1,
                "step": 0.01,
                "mode": "slider"
              }
            },
            "name": "Level",
            "description": "The volume. 0 is inaudible, 1 is the maximum volume."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                4
              ]
            }
          ]
        }
      },
      "volume_mute": {
        "name": "Mute/unmute volume",
        "description": "Mutes or unmutes the media player.",
        "fields": {
          "is_volume_muted": {
            "required": true,
            "selector": {
              "boolean": {}
            },
            "name": "Muted",
            "description": "Defines whether or not it is muted."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                8
              ]
            }
          ]
        }
      },
      "media_seek": {
        "name": "Seek",
        "description": "Allows you to go to a different part of the media that is currently playing.",
        "fields": {
          "seek_position": {
            "required": true,
            "selector": {
              "number": {
                "min": 0,
                "max": 9223372036854776000,
                "step": 0.01,
                "mode": "box"
              }
            },
            "name": "Position",
            "description": "Target position in the currently playing media. The format is platform dependent."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                2
              ]
            }
          ]
        }
      },
      "join": {
        "name": "Join",
        "description": "Groups media players together for synchronous playback. Only works on supported multiroom audio systems.",
        "fields": {
          "group_members": {
            "required": true,
            "example": "- media_player.multiroom_player2\n- media_player.multiroom_player3\n",
            "selector": {
              "entity": {
                "multiple": true,
                "domain": [
                  "media_player"
                ],
                "reorder": false
              }
            },
            "name": "Group members",
            "description": "The players which will be synced with the playback specified in 'Targets'."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                524288
              ]
            }
          ]
        }
      },
      "select_source": {
        "name": "Select source",
        "description": "Sends the media player the command to change input source.",
        "fields": {
          "source": {
            "required": true,
            "example": "video1",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Source",
            "description": "Name of the source to switch to. Platform dependent."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                2048
              ]
            }
          ]
        }
      },
      "select_sound_mode": {
        "name": "Select sound mode",
        "description": "Selects a specific sound mode.",
        "fields": {
          "sound_mode": {
            "example": "Music",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Sound mode",
            "description": "Name of the sound mode to switch to."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                65536
              ]
            }
          ]
        }
      },
      "play_media": {
        "name": "Play media",
        "description": "Starts playing specified media.",
        "fields": {
          "media": {
            "required": true,
            "selector": {
              "media": {}
            },
            "example": "{\"media_content_id\": \"https://home-assistant.io/images/cast/splash.png\", \"media_content_type\": \"music\"}",
            "name": "Media",
            "description": "The media selected to play."
          },
          "enqueue": {
            "filter": {
              "supported_features": [
                2097152
              ]
            },
            "required": false,
            "selector": {
              "select": {
                "options": [
                  "play",
                  "next",
                  "add",
                  "replace"
                ],
                "translation_key": "enqueue",
                "custom_value": false,
                "sort": false,
                "multiple": false
              }
            },
            "name": "Enqueue",
            "description": "If the content should be played now or be added to the queue."
          },
          "announce": {
            "filter": {
              "supported_features": [
                1048576
              ]
            },
            "required": false,
            "example": "true",
            "selector": {
              "boolean": {}
            },
            "name": "Announce",
            "description": "If the media should be played as an announcement."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                512
              ]
            }
          ]
        }
      },
      "browse_media": {
        "name": "Browse media",
        "description": "Browses the available media.",
        "fields": {
          "media_content_type": {
            "required": false,
            "example": "music",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Content type",
            "description": "The type of the content to browse, such as image, music, TV show, video, episode, channel, or playlist."
          },
          "media_content_id": {
            "required": false,
            "example": "A:ALBUMARTIST/Beatles",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Content ID",
            "description": "The ID of the content to browse. Integration dependent."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                131072
              ]
            }
          ]
        },
        "response": {
          "optional": false
        }
      },
      "search_media": {
        "name": "Search media",
        "description": "Searches the available media.",
        "fields": {
          "search_query": {
            "required": true,
            "example": "Beatles",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Search query",
            "description": "The term to search for."
          },
          "media_content_type": {
            "required": false,
            "example": "music",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Content type",
            "description": "The type of the content to browse, such as image, music, TV show, video, episode, channel, or playlist."
          },
          "media_content_id": {
            "required": false,
            "example": "A:ALBUMARTIST/Beatles",
            "selector": {
              "text": {
                "multiline": false,
                "multiple": false
              }
            },
            "name": "Content ID",
            "description": "The ID of the content to browse. Integration dependent."
          },
          "media_filter_classes": {
            "required": false,
            "example": [
              "album",
              "artist"
            ],
            "selector": {
              "text": {
                "multiple": true,
                "multiline": false
              }
            },
            "name": "Media class filter",
            "description": "List of media classes to filter the search results by."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                4194304
              ]
            }
          ]
        },
        "response": {
          "optional": false
        }
      },
      "shuffle_set": {
        "name": "Set shuffle",
        "description": "Enables or disables the shuffle mode.",
        "fields": {
          "shuffle": {
            "required": true,
            "selector": {
              "boolean": {}
            },
            "name": "Shuffle mode",
            "description": "Whether the media should be played in randomized order or not."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                32768
              ]
            }
          ]
        }
      },
      "unjoin": {
        "name": "Unjoin",
        "description": "Removes the player from a group. Only works on platforms which support player groups.",
        "fields": {},
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                524288
              ]
            }
          ]
        }
      },
      "repeat_set": {
        "name": "Set repeat",
        "description": "Sets the repeat mode.",
        "fields": {
          "repeat": {
            "required": true,
            "selector": {
              "select": {
                "options": [
                  "off",
                  "all",
                  "one"
                ],
                "translation_key": "repeat",
                "custom_value": false,
                "sort": false,
                "multiple": false
              }
            },
            "name": "Repeat mode",
            "description": "Whether the media (one or all) should be played in a loop or not."
          }
        },
        "target": {
          "entity": [
            {
              "domain": [
                "media_player"
              ],
              "supported_features": [
                262144
              ]
            }
          ]
        }
      }
    }
  }
"""
# 5. 优化提取指令（精准适配Media Player实体）
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""从以下homeassitant实体和对应的service中提取相应的Home Assistant实体事实性信息：

【entity】
{demo_entity}
【service】
{demo_service}
"""
        }
    ]
})

# 6. 解析并输出提取结果（适配EntityFact字段）
entity_fact = result["structured_response"]
print(entity_fact)
print("=== 提取的HA Media Player实体事实性信息 ===")
# 遍历每个实体（此处为小米AI音箱单实体）
print(f"  HA实体ID：{entity_fact.entity_id}")
print(f"  友好名称：{entity_fact.friendly_name}")
print(f"  可采集状态类型：{entity_fact.states}")
print(f"  支持的操作能力（HA Services）：{entity_fact.capabilities}")
print(f"  实体匹配线索：{entity_fact.entity_matching_clues}")
print(f"  其他补充信息：{entity_fact.others}")
