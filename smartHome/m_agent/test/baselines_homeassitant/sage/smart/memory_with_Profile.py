from langchain_core.tools import tool

from agent_project.agentcore.commons.utils import get_llm

test_preferences_str = "把灯泡和音箱分组为 “氛围” 组；记住我喜欢客厅灯是 30% 亮度；台灯太亮了，亮度调到 20% 就行；哦，睡觉时，我不喜欢黑暗的环境，这会让我害怕；还是放周杰伦的歌吧，这是我最喜欢的；我接电话时，不要让设备发出声音；哦，我房间里有卫生间；看书时，我喜欢在客厅的沙发上看，因为累了，看看客厅的植物能让我心情放松；家里的插座没有我的明确指令不能关，上面连了我的服务器；客厅灯 3000k 是让我感觉最好；我喜欢关于动物和孩子的温暖治愈故事；网关的 wifi 目前连的是我的热点（dddiu）; 你知道什么场景最吓人吗？灯一闪一闪的，然后还有恐怖背景音；准备睡前我会玩会手机，这时灯设成渐灭，30 分钟后熄灭。; 周日，我一般天亮就起；暑假的时候，我晚上都会睡在客厅沙发，然后把窗户打开，这样睡得很香。; 我一般 11 点睡觉，7 点多起床；天气好的时候，窗户要打开通风哦"
@tool
def UserProfileTool(query):
    """
    Use this to learn about the user preferences and retrieve past interactions with the user.
    Use this tool before addressing subjective or ambiguous user commands that require personalized information.
    This tool is not capable of asking the user anything.
    The query should be clear, precise and formatted as a question.
    The query should specify which type of preferences you are looking for.
    """
    import agent_project.agentcore.config.global_config as global_config
    if not global_config.ENABLE_MEMORY_FOR_TEST:
        return "记忆库中尚未存在信息"

    context = test_preferences_str
    preferences = "目前还未存在"

    prompt = f"""You are an AI who should be able to (1) infer and understand user preferences; (2) understand past interactions and extract relevant information about the user preferences.
    Try to make your best educated guess based on the available preferences and command history. Your job is to (1) First check the user preferences; (2) if the user preferences don't have the answer, you should infer the answer from the command history;

    If there is no information pertaining to the request, say so.

    The preferences of the user are: {preferences}
    The most relevant command history:
    {context}
    """.strip()

    llm = get_llm()

    system_message = {
        "role": "system",
        "content": prompt,
    }
    user_message = {
        "role": "user",
        "content": f"Question: {query}",
    }
    response = llm.invoke([system_message] + [user_message])
    return response.content