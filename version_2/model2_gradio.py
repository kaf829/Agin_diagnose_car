import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 모델 로딩
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to("cpu")

# 텍스트 생성 파이프라인
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 챗봇 응답 함수
def chat(user_input):
    prompt = (
        "당신은 자동차 수리 전문가 챗봇입니다. 차량 관련 문제에만 전문적으로 답변하세요.\n\n"
        f"### 사용자: {user_input}\n### 어시스턴트:"
    )
    output = chatbot(
    prompt,
    max_new_tokens=512,           # 토큰 더 늘리기 (덜 끊기게)
    do_sample=True,
    temperature=0.6,              # 조금 더 일관성 있게
    top_p=0.9)                    # 의미 있는 후보들 중에서만 샘플링
    result = output[0]["generated_text"].split("### 어시스턴트:")[-1].strip()
    return result

# Gradio 인터페이스
iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="사용자 질문", placeholder="예: My car won't start. What should I do?"),
    outputs=gr.Textbox(label="어시스턴트 응답"),
    title="차량 문제 챗봇 (Phi-3)",
    description="차량 문제에 대한 간단한 챗봇입니다. 질문을 입력하면 AI가 답변해줍니다.",
)

# 앱 실행
iface.launch()
