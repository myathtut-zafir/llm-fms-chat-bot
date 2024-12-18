from typing import Any, Dict, List, Set
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader,UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def run_llm(query:str,chat_history: List[Dict[str, Any]] = []):
    embeddings=OpenAIEmbeddings()

    
    new_vs_store=FAISS.load_local(folder_path='/Users/myathtut/Desktop/Code/llm-fms-chat-bot/faiss_index_react',index_name="faiss_index_react",embeddings=embeddings,allow_dangerous_deserialization=True)
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=500)
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=new_vs_store.as_retriever(), prompt=rephrase_prompt
    )
        
    qa=create_retrieval_chain(retriever=history_aware_retriever,combine_docs_chain=combine_docs)
    
    result=qa.invoke(input={"input":query,"chat_history":chat_history})
    new_result={
        "query":result["input"],
        "result":result["answer"],
        "source_documents":result["context"],
        }
    return new_result

if __name__=="__main__":
    res=run_llm("when can we generate'Vehicle Statement'?")
    print(res['result'])
