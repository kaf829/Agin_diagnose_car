import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 모델 ID: 성능 좋은 균형형 모델
model_id = "microsoft/Phi-3-medium-4k-instruct"

# 디바이스 설정 (GPU 있으면 자동 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",  # GPU 자동 할당
    trust_remote_code=True
)

# 텍스트 생성 파이프라인 생성
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# 응답 생성 함수
def chat(user_input):
    prompt = (
        "당신은 자동차 수리 전문가 챗봇입니다. 차량 관련 문제에만 전문적으로 친절하게 답변하세요.\n\n"
        f"### 사용자: {user_input}\n### 어시스턴트:"
    )
    output = chatbot(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    raw_text = output[0]["generated_text"]
    result = raw_text.replace(prompt, "").strip()
    return result

# Gradio UI 구성
iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="사용자 질문", placeholder="예: 시동이 안 걸려요. 어떻게 해야 하나요?"),
    outputs=gr.Textbox(label="어시스턴트 응답"),
    title="차량 문제 챗봇 (Phi-3 Medium)",
    description="자동차 고장이나 이상 현상에 대해 AI 정비사가 답변해드립니다."
)

# 앱 실행
iface.launch()
