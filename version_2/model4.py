from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 모델 ID: Llama 3 Instruct 기반 (Scout 모델은 아직 미공개)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# GPU가 있으면 GPU 사용, 없으면 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# 챗봇 파이프라인 생성
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# 챗봇 응답 함수
def chat_with_bot(user_input):
    prompt = f"<|system|>당신은 자동차 정비 전문가입니다. 사용자 질문에 친절하고 정확하게 답하세요.<|user|>{user_input}<|assistant|>"
    response = chatbot(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
    print("챗봇:", response[0]["generated_text"].split("<|assistant|>")[-1].strip())

# 예시 대화
while True:
    user_input = input("사용자: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chat_with_bot(user_input)
