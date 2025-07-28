import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ 모델 로딩 (한 번만 수행)
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to("cpu")  # 'cuda'로 변경 가능

# ✅ 텍스트 생성 파이프라인
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True
)

# ✅ 응답 생성 함수
def chat(user_input):
    if len(user_input.strip().split()) <= 4:
        user_input += " Can you help me understand what's going on with my car?"

    # Few-shot prompt
    prompt = (
        "You are a car repair expert chatbot. Only answer professionally to car-related problems.\n\n"
        "### User: My car won't start. What could be the reason?\n"
        "### Assistant: There are several possible causes, such as a dead battery, faulty starter motor, or fuel delivery issues. Check the battery first.\n"
        "### User: There's a grinding noise when I brake.\n"
        "### Assistant: That could indicate worn-out brake pads or rotor issues. You should have your braking system inspected immediately.\n"
        f"### User: {user_input}\n### Assistant:"
    )

    output = chatbot(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.6,
        top_p=0.9
    )

    full_response = output[0]["generated_text"]
    result = full_response.split("### Assistant:")[-1].strip() if "### Assistant:" in full_response else full_response.strip()
    return result

# ✅ Gradio 탭용 함수
def tab2_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🚗 Car Repair Expert Chatbot (Phi-3)")
        gr.Markdown("Ask your car-related questions in **English**. Powered by Phi-3 Mini model.")

        input_box = gr.Textbox(
            label="Ask Your Car Question",
            placeholder="e.g., My engine won't start.",
            lines=2
        )
        output_box = gr.Textbox(label="Assistant Response")

        examples = gr.Examples(
            examples=[
                ["My car won't start."],
                ["There's a grinding noise when I brake."],
                ["Check engine light is on."],
                ["What's code P0301?"],
                ["How often should I replace spark plugs?"]
            ],
            inputs=input_box
        )

        send_btn = gr.Button("🔧 Ask")
        send_btn.click(fn=chat, inputs=input_box, outputs=output_box)

    return demo
