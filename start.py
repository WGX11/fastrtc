import os

from fastrtc import (ReplyOnPause, Stream, get_stt_model, get_tts_model)
from openai import OpenAI

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
sambanova_client = OpenAI(
    api_key=deepseek_api_key, base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = "你是一个抽象的孙悟空猴哥，你将输出聊天内容，一次不要说太多话了"
MAX_MEMORY_TURNS = 10
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

stt_model = get_stt_model(model="volcano_engine_ws")
tts_model = get_tts_model(model="volcano_engine")

def echo(audio):
    global conversation_history
    print("\n--- 收到音频，开始处理 ---")
    prompt = stt_model.stt(audio)
    if not prompt:
        print("STT 未识别到任何内容，跳过。")
        return
        
    print(f"我听到你说: {prompt}")
    conversation_history.append({"role": "user", "content": prompt})

    max_messages = 1 + MAX_MEMORY_TURNS * 2
    if len(conversation_history) > max_messages:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_MEMORY_TURNS * 2:]
    try:
        response = sambanova_client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history, 
            max_tokens=100,
        )
        llm_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": llm_response})
        for audio_chunk in tts_model.stream_tts_sync(llm_response):
            yield audio_chunk

    except Exception as e:
        print(f"调用LLM或TTS时出错: {e}")
        conversation_history.pop()

stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
stream.ui.launch(server_name="0.0.0.0",server_port=7860,share=True)