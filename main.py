from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader,UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


# llm=ChatOllama(temperature=0, model="llama3.2", max_tokens=500)
if __name__=='__main__':
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
    
    result=retrieval_chain.invoke(input={"input":"explain me the 'Application Submission to Settlement' flow."})
    print(result["answer"])


