import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke("What is the capital of India?")
print(response.content)