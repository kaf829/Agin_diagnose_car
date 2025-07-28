from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import torch

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")  # CPU 명시

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

def car_repair_bot(user_input):
    prompt = f"""### 사용자: {user_input}
### 자동차 정비 전문가:"""
    response = chatbot(prompt, max_new_tokens=256, temperature=0.7)
    return response[0]["generated_text"].split("### 자동차 정비 전문가:")[-1].strip()

iface = gr.Interface(
    fn=car_repair_bot,
    inputs=gr.Textbox(lines=4, label="증상 또는 질문을 입력하세요"),
    outputs=gr.Textbox(label="정비 전문가의 답변"),
    title="🚗 자동차 정비 전문가 챗봇 (phi-2)",
    description="자동차 고장 증상이나 정비 관련 질문을 입력하면 전문가처럼 답변해주는 챗봇입니다."
)

iface.launch()
