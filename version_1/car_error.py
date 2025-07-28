import os
import json
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 모델 불러오기
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face access token 필요
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ 응답 생성 함수
def generate_response(user_input):
    prompt = f"""
    [INST]
    You are a professional car mechanic.
    The user described: "{user_input.strip()}"
    Return exactly 3 possible causes and quick fixes as a JSON array.
    Format:
    [
      {{"cause": "...", "quick_fix": "..."}},
      {{"cause": "...", "quick_fix": "..."}},
      {{"cause": "...", "quick_fix": "..."}}
    ]
    Do not add extra text, only JSON.
    [/INST]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.9,
            top_p=0.85,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    try:
        data = json.loads(response)
        if not isinstance(data, list):
            data = []
    except json.JSONDecodeError:
        data = []

    while len(data) < 3:
        data.append({
            "cause": "Other possible cause",
            "quick_fix": "Inspect vehicle"
        })

    output_lines = []
    for i, item in enumerate(data[:3], start=1):
        cause = item.get("cause", "Unknown cause")
        quick_fix = item.get("quick_fix", "No suggestion")
        output_lines.append(f"{i}. {cause} → {quick_fix}")

    return "\n".join(output_lines)

# ✅ 스타일 포함한 탭 함수
def tab1_ui():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
    }
    h1 {
        font-size: 2.2em;
        text-align: center;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .description {
        text-align: center;
        font-size: 1em;
        margin-bottom: 20px;
        color: #d0d8e0;
    }
    textarea, input {
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }
    button {
        font-size: 1.1em !important;
        font-weight: bold !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        background: linear-gradient(90deg, #4facfe, #00f2fe) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        transition: transform 0.2s ease-in-out !important;
    }
    button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
    }
    .output-box {
        border-radius: 12px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        font-size: 1.05em;
        line-height: 1.6em;
        white-space: pre-wrap;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 25px;
        color: #ccd5e0;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Column():
            gr.HTML("<h1>🚗 자동차 상담 챗봇</h1>")
            gr.HTML("<p class='description'>자동차 문제의 3가지 원인과 해결책을 빠르게 안내합니다.</p>")
            user_input = gr.Textbox(lines=4, placeholder="차량 증상을 입력하세요...", label="차량 증상 입력")
            submit_btn = gr.Button("🔍 진단하기")
            output_box = gr.Textbox(label="진단 결과", elem_classes=["output-box"])
            gr.HTML("<footer>© 2025 Car Diagnosis Bot | Premium Blue Theme</footer>")
        submit_btn.click(generate_response, inputs=user_input, outputs=output_box)
    return demo
