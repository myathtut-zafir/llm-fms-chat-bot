from typing import Set
from dotenv import load_dotenv

load_dotenv()
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


def ingest_docs():
    pdf_path="/Users/myathtut/Desktop/Code/llm-fms-chat-bot/FMS MY.pdf"
    loader=PyPDFLoader(file_path=pdf_path)
    excelLoader = UnstructuredExcelLoader("/Users/myathtut/Desktop/Code/llm-fms-chat-bot/FMS_MY.xlsx")
    documents=loader.load()
    excelLoad=excelLoader.load()
        
    all_documents = excelLoad + documents
        
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=all_documents)
        
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embedding=embeddings)
    vectorstore.save_local('faiss_index_react')
        
    FAISS.load_local("faiss_index_react",embeddings=embeddings,allow_dangerous_deserialization=True)

if __name__=="__main__":
    ingest_docs()