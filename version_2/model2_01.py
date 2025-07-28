import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 모델 ID 설정 (128k context 지원하는 중간 크기 모델)
model_id = "microsoft/Phi-3-medium-128k-instruct"

# GPU 사용 설정 (CUDA 사용 가능할 경우)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 토크나이저 및 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # GPU 사용 시 메모리 효율
    device_map="auto",           # 자동으로 GPU에 올려줌
    trust_remote_code=True
)

# 텍스트 생성 파이프라인 구성
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 프롬프트 입력
prompt = "### 사용자: 시동이 안 걸려요. 어떻게 해야 하나요?\n### 어시스턴트:"

# 응답 생성
output = chatbot(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)

# 결과 출력
print(output[0]["generated_text"])
