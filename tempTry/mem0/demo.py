api_key="m0-trb5hbCbDvpNEoF9awokupVT8ojQvR8VTPKcoL2A"

from mem0 import MemoryClient

client = MemoryClient(api_key=api_key)

messages = [
  { "role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts." },
  { "role": "assistant", "content": "Hello Alex! I see that you're a vegetarian with a nut allergy." }
]

client.add(messages, user_id="alex")

query = "What do you know about me?"

filters = {
 "OR":[
    {
       "user_id":"alex"
    }
 ]
}

print(client.search(query, version="v2", filters=filters))