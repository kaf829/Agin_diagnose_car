# -*- coding: utf-8 -*-
import os
import io
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import streamlit as st

# ===============================
# 1. Claude API
# ===============================
load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# ===============================
# 2. PDF → 텍스트 추출 (pdfplumber + OCR)
# ===============================
def extract_pdf_to_text(file):
    text = ""
    pdf_bytes = file.read()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                # OCR fallback
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img, lang="kor+eng")
                    text += ocr_text + "\n"
    # 로그 출력: 텍스트 길이와 키워드 유무
    st.write(f"[DEBUG] 추출 텍스트 길이: {len(text)}")
    if "건전지" in text:
        st.write("🔍 '건전지' 키워드가 추출된 텍스트에 포함됨!")
    return text

# ===============================
# 3. 텍스트 청크 분할
# ===============================
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ===============================
# 4. ChromaDB 인덱스 생성
# ===============================
def build_chroma_index(pdf_files, embedding_model):
    chroma_client = chromadb.Client(Settings())
    # 기존 manual 컬렉션 삭제
    for c in chroma_client.list_collections():
        if c.name == "manual":
            chroma_client.delete_collection("manual")
    collection = chroma_client.create_collection("manual")

    for pdf in pdf_files:
        text = extract_pdf_to_text(pdf)
        if not text.strip():
            st.warning(f"{pdf.name}에서 텍스트 추출 실패")
            continue
        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            collection.add(
                ids=[f"{pdf.name}_{i}"],
                embeddings=[emb],
                documents=[chunk]
            )
    return collection

# ===============================
# 5. 검색 (질문 키워드 기반 + 벡터 혼합)
# ===============================
def keyword_filter(question, docs):
    """간단 키워드 기반 필터링"""
    keywords = re.findall(r"[가-힣A-Za-z0-9]+", question)
    hits = [d for d in docs if any(k in d for k in keywords)]
    return hits

def search_context(question, collection, embedding_model, top_k=3):
    q_emb = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k*2)  # 조금 더 넓게 검색
    docs = results["documents"][0]
    # 키워드 기반 추가 필터링
    keyword_hits = keyword_filter(question, docs)
    final_docs = keyword_hits if keyword_hits else docs[:top_k]
    return "\n---\n".join(final_docs)

# ===============================
# 6. Claude 응답
# ===============================
def ask_claude(question, collection, embedding_model):
    context = search_context(question, collection, embedding_model)
    prompt = f"""
너는 자동차 설명서 전문가다.
아래는 Owner's Manual에서 가져온 관련 내용이다:
--------------------
{context}
--------------------
질문: {question}

설명서 내용 기반으로만 답변하되, 내용이 부족하면
'설명서에 직접적인 언급은 없지만 유사한 내용이 있습니다'라고 말하고,
가능한 관련 정보를 요약해줘.
"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# ===============================
# 7. Streamlit UI
# ===============================
def main():
    st.title("Claude + Owner's Manual (정확도 강화 버전)")

    uploaded_files = st.file_uploader("PDF 파일 업로드 (여러 개 가능)", type="pdf", accept_multiple_files=True)
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        if not uploaded_files:
            st.warning("PDF 파일을 업로드하세요.")
            return

        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        with st.spinner("PDF 분석 및 벡터 DB 생성 중..."):
            collection = build_chroma_index(uploaded_files, embedding_model)

        with st.spinner("Claude 응답 생성 중..."):
            answer = ask_claude(question, collection, embedding_model)

        st.subheader("Claude 응답")
        st.write(answer)

if __name__ == "__main__":
    main()
