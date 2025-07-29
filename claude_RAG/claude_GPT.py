import os
from dotenv import load_dotenv
import anthropic

# ✅ 환경변수 로드
load_dotenv()

# ✅ 시스템 프롬프트
SYSTEM_PROMPT = (
  ""
)

# ✅ Claude 클라이언트 설정
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# ✅ Claude 호출 함수 정의
def call_claude(user_question: str) -> str:
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        system=SYSTEM_PROMPT,  # ✅ system 프롬프트는 별도 파라미터로 전달
        messages=[
            {"role": "user", "content": user_question}
        ]
        )


    return response.content[0].text

# ✅ 예시 사용
if __name__ == "__main__":
    user_input = input("🚗 사용자 질문: ")
    answer = call_claude(user_input)
    print("\n🛠️ Claude 응답:")
    print([answer])
                                                                                                                                                                                                                      