import streamlit as st
import os
import re
from retriever_claude import ingest_pdf, ask_with_context_claude, ask_across_collections_claude
from utils import get_file_hash, vectorstore_exists
from datetime import datetime

st.set_page_config(page_title="🚗 차량 매뉴얼 Claude GPT", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #2F80ED;'>📘 현대차 매뉴얼 기반 Claude QA 챗봇</h1>
    <p style='text-align: center; font-size: 18px;'>차량 고장 증상이나 조작법에 대해 질문해보세요!<br>업로드한 매뉴얼 또는 전체 문서에서 정확한 정보를 제공합니다.</p>
    <hr style='border-top: 3px solid #bbb;'>
""", unsafe_allow_html=True)

# ✅ 상태 초기화
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ✅ 벡터 디렉토리 스캔해서 기존 컬렉션 목록 가져오기
def list_collections():
    dirs = [d for d in os.listdir("./vectorstore") if os.path.isdir(os.path.join("./vectorstore", d))]
    return sorted(dirs)

existing_collections = list_collections()

# ✅ 기존 저장된 벡터 선택 UI (선택 안 해도 가능)
selected_collection = None
if existing_collections:
    selected_collection = st.selectbox("📁 특정 매뉴얼 선택 (선택하지 않으면 전체에서 검색)", ["전체 매뉴얼 검색"] + existing_collections)
    if selected_collection != "전체 매뉴얼 검색":
        st.session_state["collection_name"] = selected_collection
    else:
        st.session_state["collection_name"] = None

# ✅ PDF 업로드 처리
with st.expander("📄 새로운 차량 매뉴얼 업로드 (선택)"):
    uploaded = st.file_uploader("PDF 파일 선택", type="pdf")
    if uploaded:
        with st.spinner("📥 PDF 업로드 중..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded.read())

            file_hash = get_file_hash("temp.pdf")[:8]

            # 1. 파일명에서 확장자 제거 후 특수문자, 한글 제거
            raw_name = os.path.splitext(uploaded.name)[0]

            # 2. 한글 및 특수문자 제거 (ASCII 문자만 유지)
            clean_name = re.sub(r"[^\x00-\x7F]", "", raw_name)       # 한글 등 비ASCII 제거
            clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_name)  # 허용 문자만 유지
            clean_name = clean_name.strip("._-")                     # 시작/끝 문자 정리

            # 3. 길이 제한 (최대 30자)s
            short_name = clean_name[:30] if len(clean_name) >= 3 else f"doc_{file_hash}"

            # 4. 최종 컬렉션 이름
            collection_name = f"{short_name}_{file_hash}"

            if not vectorstore_exists(collection_name):
                with st.spinner("🧠 문서 임베딩 중입니다..."):
                    ingest_pdf("temp.pdf", collection_name)
                st.success("✅ 문서 임베딩 완료!")
            else:
                st.info("📁 이미 업로드된 문서입니다. 벡터스토어 불러옵니다.")

            os.remove("temp.pdf")
            st.session_state["collection_name"] = collection_name
            existing_collections = list_collections()

# ✅ 채팅 UI
st.markdown("<h3>💬 정비사와의 채팅 (Claude)</h3>", unsafe_allow_html=True)
user_input = st.chat_input("🚗 궁금한 점이나 고장 증상을 입력해 주세요...")

if user_input:
    with st.spinner("🔍 정비사 응답 생성 중 (Claude)..."):
        if st.session_state.get("collection_name"):
            result = ask_with_context_claude(user_input, st.session_state["collection_name"])
        else:
            result = ask_across_collections_claude(user_input)

        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["result"]
        })

# ✅ 채팅 기록 출력
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ 대화 로그 저장
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("💾 대화 로그 저장"):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/log_{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                f.write(f"[USER] {msg['content']}\n")
            elif msg["role"] == "assistant":
                f.write(f"[ASSISTANT] {msg['content']}\n\n")
    st.success(f"💾 로그 저장 완료: {log_path}")
