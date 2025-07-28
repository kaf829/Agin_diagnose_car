# -*- coding: utf-8 -*-
import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import streamlit as st

# ===============================
# 1. Claude API 키 로드
# ===============================
load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# ===============================
# 2. PDF → 텍스트 추출
# ===============================
def extract_pdf_to_text(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ===============================
# 3. 텍스트 청크 분할
# ===============================
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ===============================
# 4. 여러 PDF → ChromaDB 인덱스 생성
# ===============================
def build_chroma_index(pdf_files, embedding_model):
    chroma_client = chromadb.Client(Settings())
    # 기존 컬렉션 있으면 삭제
    for c in chroma_client.list_collections():
        if c.name == "manual":
            chroma_client.delete_collection("manual")
    collection = chroma_client.create_collection("manual")

    for pdf in pdf_files:
        text = extract_pdf_to_text(pdf)
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
# 5. 질문 → 검색
# ===============================
def search_context(question, collection, embedding_model, top_k=3):
    q_emb = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    return "\n---\n".join(results["documents"][0])

# ===============================
# 6. Claude 응답
# ===============================
def ask_claude(question, collection, embedding_model):
    context = search_context(question, collection, embedding_model)
    if not context.strip():
        return "설명서에 관련 정보가 없습니다."
    prompt = f"""
너는 자동차 설명서 전문가다.
다음은 Owner's Manual 일부 내용이다:
--------------------
{context}
--------------------
질문: {question}

반드시 위 설명서 내용에 기반해서만 답변하라.
"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# ===============================
# 7. Streamlit UI
# ===============================
def main():
    st.title("Claude + Owner's Manual 기반 Q&A")

    uploaded_files = st.file_uploader("PDF 파일 업로드 (여러 개 선택 가능)", type="pdf", accept_multiple_files=True)
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        if not uploaded_files:
            st.warning("PDF 파일을 업로드하세요.")
            return

        with st.spinner("임베딩 모델 로드 중..."):
            embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        with st.spinner("PDF 분석 및 ChromaDB 인덱스 생성 중..."):
            collection = build_chroma_index(uploaded_files, embedding_model)

        with st.spinner("Claude 응답 생성 중..."):
            answer = ask_claude(question, collection, embedding_model)
        
        st.subheader("Claude 응답")
        st.write(answer)

if __name__ == "__main__":
    main()
