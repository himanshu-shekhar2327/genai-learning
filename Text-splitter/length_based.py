# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# text = """Artificial Intelligence is transforming industries.
# Machine Learning is a subset of AI.
# Deep Learning is a subset of Machine Learning.
# Transformers are neural networks.
# Large Language Models use transformers.
# """

# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=""
# )

# result = splitter.split_text(text)

# print(result)


# splitter = CharacterTextSplitter(
#     chunk_size =30,
#     chunk_overlap=0,
#     separator=''
# )

# result = splitter.split_text(text)

# print(result[0])
# print()
# print(result)
# print(result[0].page_content())


from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Text-splitter/dl-curriculum.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_documents(docs)

print(result[0])

