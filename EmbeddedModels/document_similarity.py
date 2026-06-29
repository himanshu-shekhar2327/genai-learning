from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-2-preview',dimensions = 300)

documents = [
    "Delhi is the capital city of India. It is home to important government buildings like the Parliament and Rashtrapati Bhavan.",

    "Mumbai is the financial capital of India. It is famous for Bollywood, the Gateway of India, and its busy seaport.",

    "Cricket is the most popular sport in India. Millions of people watch international matches and support their favorite teams.",

    "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data and make predictions without being explicitly programmed.",

    "The Taj Mahal is a UNESCO World Heritage Site located in Agra. It was built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal."
]

query = 'Tell me about Delhi'

documents_embedding = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],documents_embedding)[0]
 
index , scores = sorted(list(enumerate(scores)),key = lambda x : x[1])[-1]

print(query)
print(documents[index])
print('similarity score is :',scores)