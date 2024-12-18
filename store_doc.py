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
    pdf_path_2="/Users/myathtut/Desktop/Code/llm-fms-chat-bot/Calculation Logic [MY].pdf"
    loader=PyPDFLoader(file_path=pdf_path)
    loader2=PyPDFLoader(file_path=pdf_path_2)
    excelLoader = UnstructuredExcelLoader("/Users/myathtut/Desktop/Code/llm-fms-chat-bot/FMS_MY.xlsx")
    documents=loader.load()
    documents2=loader2.load()
    excelLoad=excelLoader.load()
        
    all_documents = excelLoad + documents+documents2
        
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=all_documents)
        
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embedding=embeddings)
    vectorstore.save_local(folder_path='/Users/myathtut/Desktop/Code/llm-fms-chat-bot/faiss_index_react',index_name="faiss_index_react")
if __name__=="__main__":
    ingest_docs()