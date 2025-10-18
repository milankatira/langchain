from dotenv import load_dotenv
from langchain_community.llms import Ollama

load_dotenv()

llm = Ollama(model="llama3")
result = llm.invoke("What is the capital of India?")
print(result)
