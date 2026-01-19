import json

import requests


def init_env():
    """
    :return:
    """
    from smartHome.m_agent.memory.fake.fake_request import HOMEASSITANT_DATA
    HOMEASSITANT_DATA.init_entities()
    tdel=-1

