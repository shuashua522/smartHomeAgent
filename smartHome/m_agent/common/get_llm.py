from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManager

from smartHome.m_agent.common.global_config import GLOBALCONFIG


def get_llm():
    # max_tokens: int = 1000
    provider=GLOBALCONFIG.provider
    model = GLOBALCONFIG.model
    base_url = GLOBALCONFIG.base_url
    api_key = GLOBALCONFIG.api_key

    llm = init_chat_model(
        model=model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        # max_tokens=max_tokens  # 配置max_tokens
    )
    return llm