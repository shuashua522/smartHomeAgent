import json

import requests

from agent_project.agentcore.config.global_config import HOMEASSITANT_AUTHORIZATION_TOKEN, HOMEASSITANT_SERVER, \
    ACTIVE_PROJECT_ENV, PRIVACYHANDLER
from agent_project.agentcore.smart_home_agent.fake_request.fake_do_service import \
    fake_execute_domain_service_by_entity_id

token = HOMEASSITANT_AUTHORIZATION_TOKEN
server = HOMEASSITANT_SERVER
active_project_env = ACTIVE_PROJECT_ENV
privacyHandler = PRIVACYHANDLER
def execute(domain,service,body):
    """
    Calls a service within a specific domain. Will return when the service has been executed.

    由于智能家居的数据已经进行加密处理(加密后的数据形如：@xxx@)，如果你需要对传入body中的某些加密数据进行算术运算。你可以用在其前后加入算术运算，例如：
    {"entity_id": "@nB/MRO8IqOyD9Kj8t9A3kw==:5sWFd4t1UNtxvhX2LYYaqOZ6aVIKfXw7LiBwXmE/d38n30HHZColHIGWTZPpQlo6@", "brightness_pct": @n+4XiEGjo3K4qp1+WdooLw==:E034U68+xYq6U47e5i/isA==@*5-4}

    """
    return fake_execute_domain_service_by_entity_id(domain, service, body)
    # import agent_project.agentcore.config.global_config as global_config
    # logger = global_config.GLOBAL_AGENT_DETAILED_LOGGER
    # if logger != None:
    #     logger.info("\n请求的body:\n" + body)
    # result = None
    # if active_project_env == "dev":
    #     headers = {
    #         "Authorization": f"Bearer {token}",
    #         "Content-Type": "application/json"
    #     }
    #
    #     url = f"http://{server}/api/services/{domain}/{service}"
    #     # 设置请求体数据
    #     # payload = {
    #     #     "entity_id": entity_id
    #     # }
    #     payload = json.loads(body)
    #
    #     # 发送POST请求
    #     response = requests.post(
    #         url=url,
    #         json=payload,  # 自动将字典转换为JSON并设置正确的Content-Type
    #         headers=headers
    #     )
    #     # 检查请求是否成功
    #     response.raise_for_status()
    #     # 返回JSON响应
    #     result = response.json()
    #
    # return result
def case_01_env():
    """
    - 插座：关闭
    - 灯泡：打开，
    - 台灯：关闭，
    - 人体传感器：放在包装里，不用打开
    - 门窗传感器：放在包装里，不用打开
    - 网关：正常
    - 音箱：播放音乐
    :return:
    """
    turn_off_plug()
    turn_on_bulb(40, 4000)
    turn_off_desk_lamp()
    send_speaker_command("播放晴天")
    send_speaker_command("音量调到8%")

def case_02_env():
    """
    - 插座：关闭
    - 灯泡：打开，冷色
    - 台灯：打开，
    - 人体传感器：放在包装里，不用打开
    - 门窗传感器：放在包装里，不用打开
    - 网关：正常，灯关掉
    - 音箱：播放音乐,音量低
    :return:
    """
    turn_off_plug()
    turn_on_bulb(40, 5700)
    turn_on_desk_lamp(40)
    send_speaker_command("播放晴天")
    send_speaker_command("音量调到8%")

def case_03_env():
    """
    - 音箱：播放音乐,音量中
    :return:
    """
    send_speaker_command("播放晴天")
    send_speaker_command("音量调到8%")

def send_speaker_command(command:str):
    """
    音箱执行命令
    :return:
    """
    domain="notify"
    service="send_message"
    body_dict = {"entity_id": "notify.xiaomi_cn_701074704_l15a_execute_text_directive_a_7_4",
                 "message": f"[{command},true]"}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain,service,body)

def turn_on_bulb(brightness,color_temp):
    """
    brightness:百分比
    打开灯泡
    :return:
    """
    domain = "light"
    service = "turn_on"
    body_dict = {"entity_id":"light.yeelink_cn_1162511951_mbulb3_s_2","brightness_pct":brightness,"color_temp_kelvin":color_temp}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)

def turn_off_bulb():
    """
    关闭灯泡
    :return:
    """
    domain = "light"
    service = "turn_off"
    body_dict = {"entity_id": "light.yeelink_cn_1162511951_mbulb3_s_2"}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)

def turn_on_desk_lamp(brightness):
    """
    打开台灯
    :return:
    """
    domain = "light"
    service = "turn_on"
    body_dict = {"entity_id": "light.philips_cn_1061200910_lite_s_2", "brightness_pct": brightness}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)
def turn_off_desk_lamp():
    """
    关闭台灯
    :return:
    """
    domain = "light"
    service = "turn_off"
    body_dict = {"entity_id": "light.philips_cn_1061200910_lite_s_2"}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)

def turn_on_plug():
    """
    打开插座
    :return:
    """
    domain = "switch"
    service = "turn_on"
    body_dict = {"entity_id": "switch.cuco_cn_269067598_cp1_on_p_2_1"}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)
def turn_off_plug():
    """
    关闭插座
    :return:
    """
    domain = "switch"
    service = "turn_off"
    body_dict = {"entity_id": "switch.cuco_cn_269067598_cp1_on_p_2_1"}
    # 将字典转为JSON字符串
    body = json.dumps(body_dict)
    execute(domain, service, body)

def enable_test_memory():
    """
    启用测试记忆
    :return:
    """
    import agent_project.agentcore.config.global_config as global_config
    global_config.ENABLE_MEMORY_FOR_TEST=True;

def disable_test_memory():
    """
    关闭测试记忆
    :return:
    """
    import agent_project.agentcore.config.global_config as global_config
    global_config.ENABLE_MEMORY_FOR_TEST = False;