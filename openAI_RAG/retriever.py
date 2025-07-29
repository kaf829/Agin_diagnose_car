from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# ✅ 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 시스템 프롬프트
SYSTEM_PROMPT = (
    "당신은 자동차 시스템 도우미 입니다. "
    "운전자가 입력한 것이 고장 증상일 경우 증상에 대해 가능한 원인을 추정하고, "
    "점검 방법과 조치 방법을 구체적이고 친절하게 설명해 주세요. "
    "정확하지 않다면 전문가의 점검이 필요하다고 안내해 주세요. "
    "운전자가 입력한 것이 고장 증상이 아닌 조작법 혹은 안내사항 주의 사항일 경우 "
    "해당 내용을 친절하게 설명해 주세요. 잘 모르겠으면 해당 답변은 잘 모르겠습니다라고 말해주세요."
)

# ✅ 1. PDF 임베딩 및 저장 함수

def ingest_pdf(pdf_path: str, collection_name: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=f"./vectorstore/{collection_name}"
    )

    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        vectordb.add_documents(batch)

    vectordb.persist()

# ✅ 2. 단일 컬렉션 질문 응답

def ask_with_context(question: str, collection_name: str, top_k: int = 5) -> dict:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=f"./vectorstore/{collection_name}",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"문서 내용:\n{context}"),
        HumanMessage(content=f"질문: {question}")
    ]

    chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    answer = chat(messages)

    return {
        "result": answer.content,
        "source_documents": docs
    }

# ✅ 3. 전체 벡터스토어에서 질문 응답

def ask_across_collections(question: str, vectorstore_root: str = "./vectorstore", top_k: int = 5) -> dict:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    all_docs = []

    for collection_name in os.listdir(vectorstore_root):
        path = os.path.join(vectorstore_root, collection_name)
        if os.path.isdir(path):
            vectordb = Chroma(
                collection_name=collection_name,
                persist_directory=path,
                embedding_function=embeddings
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(question)
            all_docs.extend(docs)

    # 중복 제거 후 가장 유사한 top_k만 추출 (문서 길이로 단순 정렬)
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    top_docs = sorted(unique_docs, key=lambda d: len(d.page_content), reverse=True)[:top_k]

    context = "\n\n".join([doc.page_content for doc in top_docs])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"문서 내용:\n{context}"),
        HumanMessage(content=f"질문: {question}")
    ]

    chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    answer = chat(messages)

    return {
        "result": answer.content,
        "source_documents": top_docs
    }

# ✅ 4. 기존 체인 방식 QA (선택 사항)

def get_qa_chain(collection_name: str):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=f"./vectorstore/{collection_name}",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
