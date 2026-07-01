from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Gemini Model
model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Groq Model (FREE)
model2 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Prompt 1
prompt1 = PromptTemplate(
    template="""
Generate short and simple notes from the following text.

{text}
""",
    input_variables=["text"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="""
Generate 5 short question-answer pairs from the following text.

{text}
""",
    input_variables=["text"]
)

# Prompt 3
prompt3 = PromptTemplate(
    template="""
Merge the following notes and quiz into a well-formatted study guide.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Parallel Execution
parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

# Merge Step
merge_chain = prompt3 | model1 | parser

# Final Chain
chain = parallel_chain | merge_chain

text = """
Support Vector Machine (SVM) is a powerful supervised machine learning
algorithm used for both classification and regression tasks. It works by
finding the optimal hyperplane that separates different classes with the
maximum possible margin. The closest data points to the hyperplane are
called support vectors, and they determine the decision boundary.

SVM is highly effective for high-dimensional datasets such as text
classification, image recognition, and bioinformatics. When data is not
linearly separable, SVM uses kernel functions such as Linear, Polynomial,
and Radial Basis Function (RBF) kernels to transform the data into a
higher-dimensional space where it can be separated more easily.

Important hyperparameters include C, which controls the trade-off between
margin size and classification accuracy, and gamma, which controls the
influence of individual training examples. Although SVM is accurate and
robust, it can be computationally expensive for very large datasets.
"""

result = chain.invoke({"text": text})

print("\n========== FINAL OUTPUT ==========\n")
print(result)

print("\n========== CHAIN GRAPH ==========\n")
chain.get_graph().print_ascii()