# -*- coding: utf-8 -*-
import os
import re
import joblib
import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import streamlit as st

# ===============================
# 1. 상수 정의
# ===============================
INDEX_FAISS = "index_pymupdf.faiss"
INDEX_PKL = "index_pymupdf.pkl"

# ===============================
# 2. Claude API 키 로드
# ===============================
load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# ===============================
# 3. PDF → 텍스트 추출 (PyMuPDF)
# ===============================
def extract_pdf_to_text(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ===============================
# 4. 텍스트 청크 분할 (500단어)
# ===============================
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ===============================
# 5. 인덱스 생성
# ===============================
def build_faiss_index(pdf_files, embedding_model):
    all_chunks = []
    for pdf in pdf_files:
        text = extract_pdf_to_text(pdf)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.info(f"총 청크 수: {len(all_chunks)} → 임베딩 계산 중...")
    embeddings = embedding_model.encode(all_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, INDEX_FAISS)
    joblib.dump(all_chunks, INDEX_PKL)
    st.success("새로운 인덱스 생성 완료!")
    return index, all_chunks

# ===============================
# 6. 인덱스 로드 (캐시 사용)
# ===============================
@st.cache_resource
def load_faiss_index():
    if not os.path.exists(INDEX_FAISS) or not os.path.exists(INDEX_PKL):
        return None, None
    index = faiss.read_index(INDEX_FAISS)
    chunks = joblib.load(INDEX_PKL)
    return index, chunks

# ===============================
# 7. Hybrid 검색 (벡터 + 키워드)
# ===============================
def search_context(question, embedding_model, index, chunks, top_k=3):
    q_emb = embedding_model.encode([question])
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k * 2)
    docs = [chunks[i] for i in I[0]]

    # 키워드 매칭 우선
    keywords = re.findall(r"[가-힣A-Za-z0-9]+", question)
    keyword_hits = [doc for doc in docs if any(k in doc for k in keywords)]

    if keyword_hits:
        return "\n---\n".join(keyword_hits[:top_k])
    return "\n---\n".join(docs[:top_k])

# ===============================
# 8. Claude 응답 (요청한 프롬프트)
# ===============================
def ask_claude(question, embedding_model, index, chunks):
    context = search_context(question, embedding_model, index, chunks)
    if not context.strip():
        return "설명서에 관련 정보가 없습니다."

    prompt = f"""
당신은 자동차 시스템 도우미 입니다. 
운전자가 입력한 것이 고장 증상일 경우 증상에 대해 가능한 원인을 추정하고, 
점검 방법과 조치 방법을 구체적이고 친절하게 설명해 주세요. 
정확하지 않다면 전문가의 점검이 필요하다고 안내해 주세요. 
운전자가 입력한 것이 고장 증상이 아닌 조작법 혹은 안내사항 주의 사항일 경우 
해당 내용을 친절하게 설명해 주세요. 잘 모르겠으면 해당 답변은 잘 모르겠습니다라고 말해주세요.

자동차 설명서 일부:
--------------------
{context}
--------------------
질문: {question}

반드시 위 설명서 내용만 사용하여 답변하세요.
"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# ===============================
# 9. Streamlit UI
# ===============================
def main():
    st.title("Claude + Owner's Manual (빠른 버전)")

    # Step 1: 인덱스 생성
    uploaded_files = st.file_uploader("PDF 파일 업로드 (최초 1회)", type="pdf", accept_multiple_files=True)
    if st.button("인덱스 생성"):
        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        with st.spinner("인덱스 생성 중..."):
            build_faiss_index(uploaded_files, embedding_model)

    # Step 2: 질문 처리
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기"):
        index, chunks = load_faiss_index()
        if index is None or chunks is None:
            st.warning("인덱스가 없습니다. 먼저 인덱스를 생성하세요.")
            return
        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        with st.spinner("Claude 응답 생성 중..."):
            answer = ask_claude(question, embedding_model, index, chunks)
        st.subheader("Claude 응답")
        st.write(answer)

if __name__ == "__main__":
    main()
