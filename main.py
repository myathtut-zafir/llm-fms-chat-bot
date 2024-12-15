from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=500)
# llm=ChatOllama(temperature=0, model="llama3.2", max_tokens=500)
response = llm.invoke("Write a short story about a robot learning to feel.")
print(response.content)


