from typing import Set
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

st.header("FMS - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Loading..."):
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
        
        new_vs_store=FAISS.load_local("faiss_index_react",embeddings=embeddings,allow_dangerous_deserialization=True)

        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=500)
        
        retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
        
        retrieval_chain=create_retrieval_chain(retriever=new_vs_store.as_retriever(),combine_docs_chain=combine_docs)
        
        result=retrieval_chain.invoke(input={"input":"when can we generate'Vehicle Statement'?"})
        sources = set(
                [doc.metadata["source"] for doc in result["context"]]
            )
        formatted_response = (
                f"{result['answer']} \n\n {create_sources_string(sources)}"
            )

        print(formatted_response)
    # print(result["context"])


