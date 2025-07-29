from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.embeddings import OpenAIEmbeddings 
import anthropic
import os
from dotenv import load_dotenv
import uuid

# ✅ 환경변수 로드
load_dotenv()

# ✅ 시스템 프롬프트
SYSTEM_PROMPT = (
    "당신은 자동차 시스템 도우미 입니다. "
    "운전자가 입력한 것이 고장 증상일 경우 증상에 대해 가능한 원인을 추정하고, "
    "점검 방법과 조치 방법을 구체적이고 친절하게 설명해 주세요. "
    "정확하지 않다면 전문가의 점검이 필요하다고 안내해 주세요. "
    "운전자가 입력한 것이 고장 증상이 아닌 조작법 혹은 안내사항 주의 사항일 경우 "
    "해당 내용을 친절하게 설명해 주세요. 잘 모르겠으면 해당 답변은 잘 모르겠습니다라고 말해주세요."
)

# ✅ Claude 클라이언트 설정
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

def call_claude(messages: list) -> str:
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        messages=messages
    )
    return response.content[0].text

# ✅ PDF 임베딩

embedding_function = OpenAIEmbeddings()

def ingest_pdf(pdf_path: str, collection_name: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=f"./vectorstore/{collection_name}",
        embedding_function=embedding_function
    )

    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        vectordb.add_documents(batch)

    vectordb.persist()

# ✅ 단일 문서 기반 질문

def ask_with_context_claude(question: str, collection_name: str, top_k: int = 5) -> dict:
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=f"./vectorstore/{collection_name}",
        embedding_function=embedding_function
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n문서 내용:\n{context}\n\n질문: {question}"}
    ]
    answer = call_claude(messages)

    return {
        "result": answer,
        "source_documents": docs
    }

# ✅ 전체 문서 검색 질문

def ask_across_collections_claude(question: str, vectorstore_root: str = "./vectorstore", top_k: int = 5) -> dict:
    all_docs = []
    for collection_name in os.listdir(vectorstore_root):
        path = os.path.join(vectorstore_root, collection_name)
        if os.path.isdir(path):
            vectordb = Chroma(
                collection_name=collection_name,
                persist_directory=path,
                embedding_function=embedding_function
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(question)
            all_docs.extend(docs)

    # 중복 제거 및 정렬
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    top_docs = sorted(unique_docs, key=lambda d: len(d.page_content), reverse=True)[:top_k]

    context = "\n\n".join([doc.page_content for doc in top_docs])

    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n문서 내용:\n{context}\n\n질문: {question}"}
    ]
    answer = call_claude(messages)

    return {
        "result": answer,
        "source_documents": top_docs
    }
