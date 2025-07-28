from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math
load_dotenv()  # ← .env 파일에서 환경변수 로드

PERSIST_DIR = "./vectorstore"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 환경변수 또는 수동 전달


def ingest_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, batch_size: int = 100):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    print(f"🔍 총 문서 조각 수: {len(chunks)}")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # ✅ Chroma 빈 인스턴스 생성
    vectordb = Chroma(
        collection_name="hyundai_manual",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # ✅ batch 단위로 안전하게 add_documents 호출
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f"📦 Embedding 청크 {i} ~ {i + len(batch_chunks)}")
        vectordb.add_documents(batch_chunks)

    vectordb.persist()
def get_qa_chain():
    if not os.path.exists(PERSIST_DIR):
        raise RuntimeError("❗ 벡터 저장소가 존재하지 않습니다. 먼저 PDF를 업로드하세요.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
