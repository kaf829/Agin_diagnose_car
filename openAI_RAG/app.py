import streamlit as st
import os
from retriever import ingest_pdf, get_qa_chain

st.set_page_config(page_title="🚗 현대차 메뉴얼 QA", layout="wide")
st.title("🚘 현대자동차 메뉴얼 문서 QA 챗봇")

# ✅ PDF 업로드 처리
uploaded_pdf = st.file_uploader("PDF 메뉴얼 파일을 선택하세요 (선택)", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("PDF 임베딩 및 벡터스토어 저장 중..."):
        file_path = os.path.join("temp_uploaded.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.read())
        ingest_pdf(file_path)
        os.remove(file_path)
    st.success("✅ PDF 임베딩 완료!")

# ✅ QA 입력창
try:
    qa_chain = get_qa_chain()
    question = st.text_input("궁금한 점을 입력하세요:", placeholder="예: 스마트키 배터리 교체 방법은?")
    
    if question:
        with st.spinner("답변 생성 중..."):
            result = qa_chain(question)
            st.subheader("📘 답변")
            st.write(result['result'])

            with st.expander("🔍 참조 문서 보기"):
                for i, doc in enumerate(result['source_documents']):
                    st.markdown(f"**[문서 {i+1}]**")
                    st.markdown(doc.page_content[:500] + "...")
except RuntimeError as e:
    st.warning(str(e))
