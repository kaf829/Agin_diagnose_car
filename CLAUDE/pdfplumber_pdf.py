# -*- coding: utf-8 -*-
import os
import io
import re
import joblib
import faiss
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
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
# 2. PDF → 텍스트 추출 (페이지 진행률 + OCR 여부 표시)
# ===============================
def extract_pdf_to_text(file):
    text = ""
    pdf_bytes = file.read()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        page_progress = st.progress(0)
        for i, page in enumerate(pdf.pages):
            st.write(f"{file.name} - {i+1}/{total_pages} 페이지 처리 중...")
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                st.write(f"페이지 {i+1}: OCR 실행 중...")
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img, lang="kor+eng")
                    text += ocr_text + "\n"
            page_progress.progress(int(((i+1) / total_pages) * 100))
    return text

# ===============================
# 3. 텍스트 청크 분할 (100단어)
# ===============================
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ===============================
# 4. FAISS Index 생성 & 저장 (전체 진행률 표시)
# ===============================
def build_faiss_index(pdf_files, embedding_model, index_path="index.faiss", meta_path="index.pkl"):
    all_chunks = []
    total_files = len(pdf_files)
    overall_progress = st.progress(0)

    for file_idx, pdf in enumerate(pdf_files):
        st.write(f"파일 처리 중: {pdf.name} ({file_idx+1}/{total_files})")
        text = extract_pdf_to_text(pdf)
        if not text.strip():
            st.warning(f"{pdf.name}에서 텍스트 추출 실패")
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)

        overall_progress.progress(int(((file_idx + 1) / total_files) * 100))

    st.write("임베딩 생성 중...")
    embeddings = embedding_model.encode(all_chunks)
    st.write(f"총 청크 수: {len(all_chunks)}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, index_path)
    joblib.dump(all_chunks, meta_path)

    st.success(f"총 {total_files}개 파일 처리 완료, 청크 수: {len(all_chunks)}개")
    return index, all_chunks

# ===============================
# 5. FAISS Index 불러오기
# ===============================
def load_faiss_index(index_path="index.faiss", meta_path="index.pkl"):
    index = faiss.read_index(index_path)
    chunks = joblib.load(meta_path)
    return index, chunks

# ===============================
# 6. 검색
# ===============================
def search_context(question, embedding_model, index, chunks, top_k=3):
    query_emb = embedding_model.encode([question])
    D, I = index.search(np.array(query_emb, dtype=np.float32), top_k)
    results = [chunks[i] for i in I[0]]
    return "\n---\n".join(results)

# ===============================
# 7. Claude 응답 (프롬프트 고정)
# ===============================
def ask_claude(question, embedding_model, index, chunks):
    context = search_context(question, embedding_model, index, chunks)
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
# 8. Streamlit UI
# ===============================
def main():
    st.title("Claude + Owner's Manual (FAISS Index + 진행률 표시)")

    uploaded_files = st.file_uploader("PDF 파일 업로드 (최초 1회)", type="pdf", accept_multiple_files=True)
    question = st.text_input("질문을 입력하세요")

    if st.button("인덱스 생성(최초 1회)"):
        if not uploaded_files:
            st.warning("PDF 파일을 업로드하세요.")
            return
        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        with st.spinner("FAISS 인덱스 생성 중..."):
            build_faiss_index(uploaded_files, embedding_model)
        st.success("FAISS 인덱스 생성 완료!")

    if st.button("질문하기"):
        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        with st.spinner("FAISS 인덱스 불러오기 중..."):
            index, chunks = load_faiss_index()
        with st.spinner("Claude 응답 생성 중..."):
            answer = ask_claude(question, embedding_model, index, chunks)
        st.subheader("Claude 응답")
        st.write(answer)

if __name__ == "__main__":
    main()
