import openai
import gradio as gr

# 최신 방식: openai.Client 사용

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 자동 로드

api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

messages = [
    {"role": "system", "content": "당신은 자동차 정비사로서 고장 원인을 설명해주는 도우미입니다."}
]

def chat_with_gpt(user_input):
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    bot_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": bot_reply})
    return bot_reply

demo = gr.Interface(
    fn=chat_with_gpt,
    inputs=gr.Textbox(lines=2, placeholder="예: 시동이 안 걸려요"),
    outputs="text",
    title="자동차 고장 챗봇",
    description="GPT-3.5 기반 자동차 문제 해결 도우미"
)

demo.launch(share=True)
