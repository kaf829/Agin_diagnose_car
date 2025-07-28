import openai
import gradio as gr
import sys
import io

# ✅ 한글 인코딩 오류 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 자동 로드

api_key = os.getenv("OPENAI_API_KEY")


# ✅ OpenAI Client
client = openai.OpenAI(api_key=api_key)  # API 키는 보안상 생략 권장

# ✅ 시스템 프롬프트
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "당신은 자동차 정비 전문가입니다. "
        "운전자가 입력한 고장 증상에 대해 가능한 원인을 추정하고, "
        "점검 방법과 조치 방법을 구체적이고 친절하게 설명해 주세요. "
        "정확하지 않다면 전문가의 점검이 필요하다고 안내해 주세요."
    )
}

# ✅ 메시지 초기화
messages = [SYSTEM_PROMPT.copy()]

# ✅ 응답 생성 함수
def respond(user_input, chat_history):
    if not user_input.strip():
        return "", chat_history

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        assistant_reply = response.choices[0].message.content.strip()
        assistant_reply = assistant_reply.encode('utf-8', errors='ignore').decode('utf-8')
        messages.append({"role": "assistant", "content": assistant_reply})
        chat_history.append((user_input, assistant_reply))
        return "", chat_history
    except Exception as e:
        error_msg = f"⚠️ 오류 발생: {str(e)}"
        chat_history.append((user_input, error_msg))
        return "", chat_history

# ✅ 대화 초기화 함수
def clear_history():
    global messages
    messages = [SYSTEM_PROMPT.copy()]
    return []

# ✅ 함수로 Wrapping
def tab3_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="자동차 GPT 챗봇") as demo:
        with gr.Column(elem_id="container", scale=1):
            gr.Markdown("## 🚘 **자동차 고장 진단 GPT**", elem_id="header")
            gr.Markdown(
                "GPT-3.5가 자동차 고장 증상을 진단하고 해결 방법을 알려드립니다.\n"
                "**예시**: 시동이 안 걸려요 / 브레이크가 밀려요 / 엔진에서 소리가 나요 등",
                elem_id="subheader"
            )

            chatbot = gr.Chatbot(label="💬 GPT 자동차 정비사", height=450)

            with gr.Row():
                msg = gr.Textbox(placeholder="고장 증상을 입력하세요", label="증상 입력", scale=5)
                send_btn = gr.Button("📤 전송", scale=1)
                clear_btn = gr.Button("🧹 새 상담", scale=1)

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(clear_history, outputs=chatbot)

    return demo
