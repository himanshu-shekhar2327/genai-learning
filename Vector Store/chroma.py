from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# -----------------------------
# Embedding Model
# -----------------------------
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -----------------------------
# Create Documents
# -----------------------------
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"},
    ),
]

# -----------------------------
# Create Chroma Vector Store
# -----------------------------
vector_store = Chroma(
    collection_name="sample",
    embedding_function=embedding,
    persist_directory="my_chroma_db",
)

# -----------------------------
# Add Documents
# -----------------------------
ids = vector_store.add_documents(docs)

print("Document IDs:")
print(ids)

# -----------------------------
# View Documents
# -----------------------------
print(vector_store.get())

# -----------------------------
# Similarity Search
# -----------------------------
results = vector_store.similarity_search(
    query="Who among these are bowlers?",
    k=2
)

print("\nSimilarity Search")
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 50)

# -----------------------------
# Similarity Search with Score
# -----------------------------
results = vector_store.similarity_search_with_score(
    query="Who among these are bowlers?",
    k=2
)

print("\nSimilarity Search with Score")
for doc, score in results:
    print(score)
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 50)

# -----------------------------
# Metadata Filtering
# -----------------------------
results = vector_store.similarity_search(
    query="IPL players",
    k=5,
    filter={"team": "Chennai Super Kings"},
)

print("\nMetadata Filter")
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 50)

# -----------------------------
# Update Document
# -----------------------------
updated_doc = Document(
    page_content="""
Virat Kohli, the former captain of Royal Challengers Bangalore (RCB),
is renowned for his aggressive leadership and consistent batting performances.
He is the highest run scorer in IPL history and one of the greatest T20 batters.
""",
    metadata={"team": "Royal Challengers Bangalore"},
)

vector_store.update_document(
    document_id=ids[0],
    document=updated_doc,
)

print("\nAfter Update")
print(vector_store.get())

# -----------------------------
# Delete Document
# -----------------------------
vector_store.delete(ids=[ids[0]])

print("\nAfter Delete")
print(vector_store.get())