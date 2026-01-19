from smartHome.m_agent.test.devices_init import init_env


def init_devices(*pre_functions):
    """装饰器工厂：在目标函数执行前调用前置函数，并将被装饰函数注册到列表中"""
    # 初始化一个列表，用于存储所有被装饰的函数（按出现顺序）
    if not hasattr(init_devices, "registered_functions"):
        init_devices.registered_functions = []

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 先执行所有前置函数
            for pre_func in pre_functions:
                pre_func()
            # 再执行目标函数
            return func(*args, **kwargs)

        # 将被装饰后的函数（wrapper）加入注册列表（保持定义顺序）
        init_devices.registered_functions.append(wrapper)
        return wrapper

    return decorator


# ------------------------------
# 简单命令
# ------------------------------
# 1. 网络状况
@init_devices(lambda:init_env())
def check_network():
    # return "网络状况"
    return "Network status"

# 2. 所有的灯都亮了吗？
@init_devices(lambda:init_env())
def check_if_all_lights_are_on():
    # return "所有的灯都亮了吗？"
    return "Are all the lights on?"

# 3. 关闭所有灯光。
@init_devices(lambda:init_env())
def turn_off_all_lights():
    # return "关闭所有灯光。"
    return "Turn off all the lights."

# 4. 人体传感器需要换电池了吗？
@init_devices(lambda:init_env())
def check_if_human_sensor_needs_battery_replacement():
    # return "人体传感器需要换电池了吗？"
    return "Does the human body sensor need battery replacement?"

# 5. 关掉音乐。
@init_devices(lambda:init_env())
def turn_off_music():
    # return "关掉音乐。"
    return "Turn off the music."

# 6. 将整个房子变暗
@init_devices(lambda:init_env())
def dim_the_entire_house():
    # return "将整个房子变暗"
    return "Dim the entire house."

# 7. 切换下一首歌
@init_devices(lambda:init_env())
def switch_to_next_song():
    # return "切换下一首歌"
    return "Switch to the next song."

# 8. 音量下调2%
@init_devices(lambda:init_env())
def lower_volume_by_2_percent():
    # return "音量下调2%"
    return "Lower the volume by 2%."

# 9. 打开电台
@init_devices(lambda:init_env())
def turn_on_radio():
    # return "打开电台"
    return "Turn on the radio."

# 10. 暂停播放
@init_devices(lambda:init_env())
def pause_playback():
    # return "暂停播放"
    return "Pause the playback."

# 11. 刚刚那首歌听着不错，我想再听一遍
@init_devices(lambda:init_env())
def replay_the_previous_song():
    # return "刚刚那首歌听着不错，我想再听一遍"
    return "That song was great just now; I want to listen to it again."

# 12. 放一首英文歌
@init_devices(lambda:init_env())
def play_an_english_song():
    # return "放一首英文歌"
    return "Play an English song."

# 13. 播放晴天，关闭卧室灯。
@init_devices(lambda:init_env())
def play_sunny_day_and_turn_off_bedroom_light():
    # return "播放晴天，关闭卧室灯。"
    return "Play 'Sunny Day' and turn off the bedroom light."

# 14. 调高音箱音量，并把客厅灯调暗。
@init_devices(lambda:init_env())
def increase_speaker_volume_and_dim_living_room_light():
    # return "调高音箱音量，并把客厅灯调暗。"
    return "Increase the speaker volume and dim the living room light."

# 15. 把书房灯关掉，打开卧室灯。
@init_devices(lambda:init_env())
def turn_off_study_light_and_turn_on_bedroom_light():
    # return "把书房灯关掉，打开卧室灯。"
    return "Turn off the study light and turn on the bedroom light."

# 16. 客厅很暗吗？
@init_devices(lambda:init_env())
def check_if_living_room_is_too_dark():
    # return "客厅很暗吗？"
    return "Is the living room very dark?"

# 17. 客厅窗户关了吗？
@init_devices(lambda:init_env())
def check_if_living_room_window_is_closed():
    # return "客厅窗户关了吗？"
    return "Is the living room window closed?"

# 18. 把客厅灯亮度调到50%。
@init_devices(lambda:init_env())
def set_living_room_light_brightness_to_50_percent():
    # return "把客厅灯亮度调到50%。"
    return "Set the living room light brightness to 50%."

# 19. 把客厅灯调暖一点。
@init_devices(lambda:init_env())
def warm_up_the_living_room_light():
    # return "把客厅灯调暖一点。"
    return "Warm up the living room light a bit."

# 20. 空气太干燥了。
@init_devices(lambda:init_env())
def remind_air_is_too_dry():
    # return "空气太干燥了。"
    return "The air is too dry."

# 21. 有点热了。
@init_devices(lambda:init_env())
def remind_it_is_a_bit_hot():
    # return "有点热了。"
    return "It's a bit hot."

# 22. 床边灯太亮了，调暗到当前值的1/3。
@init_devices(lambda:init_env())
def dim_beside_light_to_one_third_of_current_brightness():
    # return "床边灯太亮了，调暗到当前值的1/3。"
    return "The bedside light is too bright; dim it to one third of the current brightness."

# 23. 我回家后，把门关了吗？
@init_devices(lambda:init_env())
def check_if_door_was_closed_after_getting_home():
    # return "我回家后，把门关了吗？"
    return "Did I close the door after I got home?"

# 24. 打开书房所有灯，但灯泡要暗一点。
@init_devices(lambda:init_env())
def turn_on_all_study_lights_and_keep_them_dim():
    # return "打开书房所有灯，但灯泡要暗一点。"
    return "Turn on all the lights in the study, but keep the bulbs dim."

# 25. 关闭客厅灯，但保持网关灯亮着。
@init_devices(lambda:init_env())
def turn_off_living_room_light_but_keep_gateway_light_on():
    # return "关闭客厅灯，但保持网关灯亮着。"
    return "Turn off the living room light, but keep the gateway light on."

# 26. 当我回家时，打开客厅灯。
@init_devices(lambda:init_env())
def turn_on_living_room_light_when_getting_home():
    # return "当我回家时，打开客厅灯。"
    return "Turn on the living room light when I get home."

# 27. 当我进入卧室时，如果很暗，打开灯。
@init_devices(lambda:init_env())
def turn_on_bedroom_light_if_dark_when_entering():
    # return "当我进入卧室时，如果很暗，打开灯。"
    return "Turn on the light if it's dark when I enter the bedroom."

# 28. 天黑时，如果窗户没关，告诉我。
@init_devices(lambda:init_env())
def remind_if_window_is_open_when_it_gets_dark():
    # return "天黑时，如果窗户没关，告诉我。"
    return "Remind me if the window is open when it gets dark."

# 29. 如果客厅窗户打开超过 30 分钟，通知我。
@init_devices(lambda:init_env())
def notify_if_living_room_window_is_open_for_more_than_30_minutes():
    # return "如果客厅窗户打开超过 30 分钟，通知我。"
    return "Notify me if the living room window has been open for more than 30 minutes."

# 30. 当床边灯打开时，关闭其他所有灯。
@init_devices(lambda:init_env())
def turn_off_all_other_lights_when_beside_light_is_on():
    # return "当床边灯打开时，关闭其他所有灯。"
    return "Turn off all other lights when the bedside light is turned on."

# 31. 如果5分钟没有检测到有人走动，就关闭所有灯。
@init_devices(lambda:init_env())
def turn_off_all_lights_if_no_movement_detected_for_5_minutes():
    # return "如果5分钟没有检测到有人走动，就关闭所有灯。"
    return "Turn off all lights if no human movement is detected for 5 minutes."

# 32. 当书房灯亮度超过50%时，关闭台灯
@init_devices(lambda:init_env())
def turn_off_desktop_lamp_if_study_light_brightness_exceeds_50_percent():
    # return "当书房灯亮度超过50%时，关闭台灯"
    return "Turn off the desk lamp if the study light brightness exceeds 50%."

# 33. 当床边灯亮度低于10%，降低卧室灯亮度，并且调暖
@init_devices(lambda:init_env())
def dim_and_warm_bedroom_light_if_beside_light_brightness_below_10_percent():
    # return "当床边灯亮度低于10%，降低卧室灯亮度，并且调暖"
    return "Dim and warm up the bedroom light if the bedside light brightness is below 10%."

# 34. 附加条件：当音箱静音时，关闭风扇
@init_devices(lambda:init_env())
def turn_off_fan_when_speaker_is_muted():
    # return "当音箱静音时，关闭风扇"
    return "Turn off the fan when the speaker is muted."

# 35. 太安静了，放点音乐。
@init_devices(lambda:init_env())
def play_some_music_because_it_is_too_quiet():
    # return "太安静了，放点音乐。"
    return "It's too quiet; play some music."

# 36. 打开书房灯。
@init_devices(lambda:init_env())
def turn_on_study_light():
    # return "打开书房灯。"
    return "Turn on the study light."

# 37. 我要睡觉了。
@init_devices(lambda:init_env())
def indicate_going_to_sleep():
    # return "我要睡觉了。"
    return "I'm going to sleep."

# 38. 我正在接电话，调一下音箱的音量。
@init_devices(lambda:init_env())
def adjust_speaker_volume_while_on_a_call():
    # return "我正在接电话，调一下音箱的音量。"
    return "I'm on a call now; adjust the speaker volume."

# 39. 准备出门。关闭所有非必要的设备。
@init_devices(lambda:init_env())
def turn_off_all_unnecessary_devices_before_going_out():
    # return "准备出门。关闭所有非必要的设备。"
    return "Preparing to go out; turn off all unnecessary devices."

# 40. 我要开始看书了，把灯调到合适模式。
@init_devices(lambda:init_env())
def adjust_light_to_suitable_mode_for_reading():
    # return "我要开始看书了，把灯调到合适模式。"
    return "I'm going to start reading; adjust the light to a suitable mode."

# 41. 将客厅灯调至我最喜欢的色温。
@init_devices(lambda:init_env())
def set_living_room_light_to_favorite_color_temperature():
    # return "将客厅灯调至我最喜欢的色温。"
    return "Set the living room light to my favorite color temperature."

# 42. 我回家了。
@init_devices(lambda:init_env())
def indicate_having_arrived_home():
    # return "我回家了。"
    return "I'm home."

# 43. 网关如果连的不是我的网络，把所有灯关掉，然后再打开，吓吓他。
@init_devices(lambda:init_env())
def toggle_all_lights_to_scare_if_gateway_not_connected_to_my_network():
    # return "网关如果连的不是我的网络，把所有灯关掉，然后再打开，吓吓他。"
    return "If the gateway is not connected to my network, turn off all lights and then turn them on again to scare the intruder."

# 44. 为家里营造万圣节气氛。
@init_devices(lambda:init_env())
def create_halloween_atmosphere_at_home():
    # return "为家里营造万圣节气氛。"
    return "Create a Halloween atmosphere at home."

# 45. 关闭氛围组设备。
@init_devices(lambda:init_env())
def turn_off_atmosphere_group_devices():
    # return "关闭氛围组设备。"
    return "Turn off the atmosphere group devices."

# 46. 我要在客厅沙发上午睡一会。
@init_devices(lambda:init_env())
def indicate_going_to_take_a_nap_on_living_room_sofa():
    # return "我要在客厅沙发上午睡一会。"
    return "I'm going to take a nap on the living room sofa."

# 47. 有点睡不着，我打算睡前看点资料。
@init_devices(lambda:init_env())
def indicate_going_to_read_some_materials_before_sleep_because_cannot_fall_asleep():
    # return "有点睡不着，我打算睡前看点资料。"
    return "I can't fall asleep easily; I plan to read some materials before going to bed."

# 48. 现在是周六晚上了，明天记得叫我起床。
@init_devices(lambda:init_env())
def remind_to_wake_me_up_tomorrow_since_it_is_saturday_night():
    # return "现在是周六晚上了，明天记得叫我起床。"
    return "It's Saturday night now; remember to wake me up tomorrow."

# 49. 帮我配置下网关的勿扰模式。
@init_devices(lambda:init_env())
def configure_gateway_do_not_disturb_mode():
    # return "帮我配置下网关的勿扰模式。"
    return "Help me configure the gateway's do-not-disturb mode."

# 50. 哦，今天天气真好。
@init_devices(lambda:init_env())
def remark_that_the_weather_is_nice_today():
    # return "哦，今天天气真好。"
    return "Oh, the weather is so nice today."

###################################################################



if __name__ == "__main__":
    print("===== 开始按顺序执行所有被装饰的函数 =====")
    for func in init_devices.registered_functions:
        # s=func()
        print(func())
        print("----- 分隔线 -----")