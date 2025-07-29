import streamlit as st
from streamlit_option_menu import option_menu
import os, re, base64
from retriever_claude import ingest_pdf, ask_with_context_claude, ask_across_collections_claude
from utils import get_file_hash, vectorstore_exists
from datetime import datetime

# ------------------------ 페이지 설정 ------------------------
st.set_page_config(page_title="📘 현대차 Claude GPT", layout="wide", page_icon="🚗")

# ------------------------ 이미지 Base64 인코딩 ------------------------
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_image_base64("hd_2.png")  # ✅ 배경 이미지 파일명

# ------------------------ 전체 배경 스타일 적용 ------------------------
st.markdown(f"""
    <style>
        body {{
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            margin: 0;
            padding: 0;
        }}

        .main {{
            display: flex;
            justify-content: center;
        }}

        .stApp {{
            background-color: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 30px;
            max-width: 1400px;
            width: calc(100% - 320px);  /* ✅ 사이드바 고려 */
            margin: 100px auto;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
        }}

        .block-container {{
            padding-bottom: 200px !important;  /* ✅ 입력창이 가려지지 않게 하단 여유 확보 */
        }}

        /* ✅ 채팅 입력창이 화면 하단에 고정되되, 사이드바를 피해서 오른쪽 정렬 */
        .stChatInputContainer {{
            position: fixed;
            bottom: 20px;
            left: 320px;   /* ✅ 사이드바 너비만큼 offset */
            right: 20px;
            z-index: 9999;
        }}

        /* ✅ 채팅 입력창 안 짤리도록 max-width 보정 */
        .st-emotion-cache-1wivap2 {{
            max-width: 100% !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------------ 세션 상태 초기화 ------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ------------------------ 사이드 메뉴 ------------------------
with st.sidebar:
   
    selected = option_menu(
        menu_title="📂 메뉴",
        options=["홈", "매뉴얼 업로드", "챗봇", ""],
        icons=["house", "file-earmark-arrow-up", "chat-left-text", ""],
        menu_icon="tools",
        default_index=0,
        styles={
            "container": {
                "padding": "10px",                # ✅ 줄임 (30px → 10px)
                "margin-top": "50px",
                "background-color": "#f0f8ff"
            },
            "icon": {"color": "blue", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "overflow-wrap": "break-word",    # ✅ 줄바꿈 허용
                "white-space": "normal"           # ✅ 공백 줄바꿈 허용
            },
            "nav-link-selected": {
                "background-color": "#1e88e5",
                "color": "white"
            }
        }
    )

# ------------------------ 홈 ------------------------
if selected == "홈":
    st.markdown("<br><br><h1 style='text-align: center; color: #0D47A1;'>🚗 현대차 매뉴얼 기반 Claude GPT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>차량 고장이나 조작법에 대해 질문하세요.</p>", unsafe_allow_html=True)
    st.markdown("### 🚗 현대차 매뉴얼 기반 Claude GPT 서비스 안내")
 
    st.markdown("#### 📘 서비스 개요")
    st.write(
        "현대차 매뉴얼 기반 Claude GPT는 현대자동차의 공식 차량 매뉴얼을 AI에 연동하여, "
        "차량 관련 질문에 정확하고 신속하게 답변해주는 지능형 정비 상담 서비스입니다. "
        "운전자는 복잡한 매뉴얼을 직접 찾을 필요 없이, 궁금한 내용을 자연어로 질문하면 "
        "Claude 모델이 매뉴얼에서 핵심 정보를 찾아 간결하게 알려줍니다."
    )
    
    st.markdown("#### 🎯 주요 기능")
    st.write("🔍 **정확한 매뉴얼 검색 응답**  \nClaude GPT는 업로드된 현대차 매뉴얼의 내용을 임베딩하여, 질문에 가장 적합한 정보를 문맥 기반으로 추출하고 답변합니다.")

    st.write("💬 **자연어 기반 대화 지원**  \n차량 고장 증상, 경고등 설명, 조작법, 유지보수 등에 대해 일상어로 질문하면 직관적인 답변을 받을 수 있습니다.")

    st.write("📄 **PDF 매뉴얼 업로드 및 관리**  \n다양한 차량 모델의 매뉴얼을 PDF 형식으로 업로드하여, 차량별로 정보를 분리하고 관리할 수 있습니다.")

# ------------------------ 매뉴얼 업로드 ------------------------
elif selected == "매뉴얼 업로드":

    st.header("📄 새로운 매뉴얼 업로드")
    uploaded = st.file_uploader("PDF 파일 선택", type="pdf")

    if uploaded:
        with st.spinner("📥 업로드 중..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded.read())

            file_hash = get_file_hash("temp.pdf")[:8]
            raw_name = os.path.splitext(uploaded.name)[0]
            clean_name = re.sub(r"[^\x00-\x7F]", "", raw_name)
            clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_name).strip("._-")
            short_name = clean_name[:30] if len(clean_name) >= 3 else f"doc_{file_hash}"
            collection_name = f"{short_name}_{file_hash}"

            if not vectorstore_exists(collection_name):
                with st.spinner("🧠 문서 임베딩 중입니다..."):
                    ingest_pdf("temp.pdf", collection_name)
                st.success(f"✅ 임베딩 완료: {collection_name}")
            else:
                st.info("📁 이미 등록된 문서입니다.")

            os.remove("temp.pdf")
            st.session_state["collection_name"] = collection_name

# ------------------------ 챗봇 ------------------------
elif selected == "챗봇":

    st.header("💬 정비사 Claude 챗봇")

    def list_collections():
        dirs = [d for d in os.listdir("./vectorstore") if os.path.isdir(os.path.join("./vectorstore", d))]
        return sorted(dirs)

    collections = list_collections()
    selected_doc = st.selectbox("📚 매뉴얼 선택 (전체 검색 가능)", ["메뉴얼 선택 필요"] + collections)
    st.session_state["collection_name"] = None if selected_doc == "메뉴얼 선택 필요" else selected_doc

    if st.session_state["collection_name"] is None:
        st.warning("📌 매뉴얼을 선택해주세요. 전체 검색 또는 특정 매뉴얼을 선택해야 질문이 가능합니다.")
        

    # ✅ 입력값 초기화는 위젯 선언 전에 수행해야 함
    if "clear_chatbox" in st.session_state and st.session_state["clear_chatbox"]:
        st.session_state["chatbox"] = ""
        st.session_state["clear_chatbox"] = False  # 초기화 완료

    # ✅ 텍스트 입력 (입력창 위쪽 배치)
    user_input = st.text_input("🚘 차량 문제나 궁금한 점을 입력하세요", key="chatbox")

    # ✅ 이전 입력 처리
    if "pending_input" in st.session_state:
        pending = st.session_state.pop("pending_input")

        with st.spinner("🔧 답변 중..."):
            if st.session_state["collection_name"]:
                result = ask_with_context_claude(pending, st.session_state["collection_name"])
            else:
                result = ask_across_collections_claude(pending)

        st.session_state.chat_messages.append({"role": "user", "content": pending})
        st.session_state.chat_messages.append({"role": "assistant", "content": result["result"]})

    # ✅ 채팅 메시지 출력
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ✅ 새로운 입력 감지
    if user_input:
        st.session_state["pending_input"] = user_input
        st.session_state["clear_chatbox"] = True  # 다음 rerun 시 초기화
        st.rerun()


# ------------------------ 대화 저장 ------------------------
# elif selected == "대화 저장":
#     st.header("💾 대화 로그 저장")
#     if st.button("📝 저장하기"):
#         os.makedirs("logs", exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         path = f"logs/log_{timestamp}.txt"
#         with open(path, "w", encoding="utf-8") as f:
#             for msg in st.session_state.chat_messages:
#                 role = msg["role"].upper()
#                 f.write(f"[{role}] {msg['content']}\n\n")
#         st.success(f"✅ 저장 완료: `{path}`")
