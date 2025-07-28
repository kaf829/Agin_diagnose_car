import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def ask_claude_only(question):
    prompt = f"""
너는 자동차 전문가야.
아래 질문에 대해 너가 이미 알고 있는 지식을 기반으로만 답변해.
질문: {question}
"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

if __name__ == "__main__":
    q = input("질문 입력: ")
    print("\n=== Claude 자체 지식 답변 ===")
    print(ask_claude_only(q))
