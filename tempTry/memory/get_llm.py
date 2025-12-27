from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManager


def get_llm():
    # max_tokens: int = 1000
    provider="doubao"
    model = "ep-20250805172515-6d5kv"
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    api_key = "5950638e-f4c0-45d0-a005-cd9331ecada8"

    llm = init_chat_model(
        model=model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        # max_tokens=max_tokens  # 配置max_tokens
    )
    return llm