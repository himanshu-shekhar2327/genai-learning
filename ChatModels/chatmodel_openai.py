import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7, max_output_tokens = 30)

# response = llm.invoke("What is the capital of India?")
response = llm.invoke("write a short story about india ")
print(response.content)