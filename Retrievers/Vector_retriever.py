from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# ----------------------------
# Step 1 : Create Documents
# ----------------------------

documents = [
    Document(
        page_content="LangChain helps developers build LLM applications easily."
    ),
    Document(
        page_content="Chroma is a vector database optimized for LLM-based search."
    ),
    Document(
        page_content="Embeddings convert text into high-dimensional vectors."
    ),
    Document(
        page_content="Google provides Gemini embedding models."
    ),
]

# ----------------------------
# Step 2 : Embedding Model
# ----------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ----------------------------
# Step 3 : Create Vector Store
# ----------------------------

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# ----------------------------
# Step 4 : Create Retriever
# ----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# ----------------------------
# Step 5 : Query Retriever
# ----------------------------

query = "What is Chroma used for?"

results = retriever.invoke(query)

print("Retriever Results\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 40)

# ----------------------------
# Step 6 : Similarity Search
# ----------------------------

print("\nSimilarity Search Results\n")

results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 40)