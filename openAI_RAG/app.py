import streamlit as st
import os
from retriever import ingest_pdf, get_qa_chain
from utils import get_file_hash, vectorstore_exists
from datetime import datetime

st.set_page_config(page_title="🚗 차량 메뉴얼 GPT", layout="wide")
st.title("📘 현대차 매뉴얼 기반 QA 챗봇")

# ✅ 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

uploaded = st.file_uploader("PDF 매뉴얼 업로드 (차량별)", type="pdf")

if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())
    file_hash = get_file_hash("temp.pdf")
    collection_name = file_hash

    if not vectorstore_exists(file_hash):
        with st.spinner("🧠 임베딩 중입니다..."):
            ingest_pdf("temp.pdf", collection_name)
        st.success("✅ 문서 임베딩 완료!")
    else:
        st.info("📁 이미 업로드된 문서입니다. 벡터스토어 불러옵니다.")

    os.remove("temp.pdf")
    st.session_state["collection_name"] = collection_name

if "collection_name" in st.session_state:
    qa_chain = get_qa_chain(st.session_state["collection_name"])

    question = st.text_input("질문을 입력하세요:")
    if question:
        result = qa_chain(question)
        st.subheader("📘 GPT 답변")
        st.write(result['result'])

        # ✅ 대화 로그 저장
        st.session_state["history"].append({
            "time": str(datetime.now()),
            "question": question,
            "answer": result["result"]
        })

        # ✅ 유사 문서 표시
        with st.expander("🔍 GPT가 참고한 문서 (최대 5개)"):
            for i, doc in enumerate(result['source_documents'][:5]):
                st.markdown(f"**문서 {i+1}**")
                st.write(doc.page_content[:500] + "...")
                st.markdown("---")

# ✅ 세션 로그 저장
if st.button("💾 대화 로그 저장"):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/log_{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for log in st.session_state["history"]:
            f.write(f"[{log['time']}]\nQ: {log['question']}\nA: {log['answer']}\n\n")
    st.success(f"💾 로그 저장 완료: {log_path}")
