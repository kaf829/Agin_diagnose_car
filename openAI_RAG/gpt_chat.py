import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "당신은 자동차 시스템 도우미 입니다. "
    "운전자가 입력한 것이 고장 증상일 경우 증상에 대해 가능한 원인을 추정하고, "
    "점검 방법과 조치 방법을 구체적이고 친절하게 설명해 주세요. "
    "정확하지 않다면 전문가의 점검이 필요하다고 안내해 주세요. "
    "운전자가 입력한 것이 고장 증상이 아닌 조작법 혹은 안내사항 주의 사항일 경우 "
    "해당 내용을 친절하게 설명해 주세요. 잘 모르겠으면 해당 답변은 잘 모르겠습니다라고 말해주세요."
)

questions = [
    "선루프 초기화 하는 방법 알려줘"
]

responses = []

for question in questions:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    answer = completion.choices[0].message.content
    responses.append(answer)

print(responses)
