# Embdedding using closed source model


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview",dimensions=32)

documents = [
    'Delhi is the capital of india',
    'Kolkata is the capital of west bengal',
    'paris is the capital of france'
]

result  =  embedding.embed_documents(documents)

print(str(result))