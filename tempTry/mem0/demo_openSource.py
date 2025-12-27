import os

from mem0 import Memory

os.environ["OPENAI_API_KEY"] = "sk-70bd7714a49d4808af6a939853fcbfce"
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "deepseek-reasoner", "temperature": 0.1,"openai_base_url":"https://api.deepseek.com/v1"},
    },
    "embedder": {
        "provider": "vertexai",
        "config": {"model": "textembedding-gecko@003"},
    },
    "reranker": {
        "provider": "cohere",
        "config": {"model": "rerank-english-v3.0"},
    },
}

m = Memory.from_config(config)


messages = [
    {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Alex! I'll remember your interests."}
]
m.add(messages, user_id="alex")

results = m.search("What do you know about me?", filters={"user_id": "alex"})
print(results)

