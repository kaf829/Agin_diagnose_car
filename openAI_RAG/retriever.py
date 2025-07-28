from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

def ingest_pdf(pdf_path: str, collection_name: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    load_dotenv()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=f"./vectorstore/{collection_name}"
    )

    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        vectordb.add_documents(batch)

    vectordb.persist()

def get_qa_chain(collection_name: str):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=f"./vectorstore/{collection_name}",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
