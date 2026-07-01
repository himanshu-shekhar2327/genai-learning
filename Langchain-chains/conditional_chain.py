from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()

# ---------------- Models ---------------- #

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

model2 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ---------------- Parsers ---------------- #

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ---------------- Prompt 1 ---------------- #

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    },
)

classifier_chain = prompt1 | model1 | parser2

# Test classifier

result = classifier_chain.invoke(
    {"feedback": "This is a wonderful smartphone."}
)

print(result)
print(result.sentiment)

# ---------------- Positive Prompt ---------------- #

prompt2 = PromptTemplate(
    template="""
Write a polite response to this positive feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

# ---------------- Negative Prompt ---------------- #

prompt3 = PromptTemplate(
    template="""
Write a polite response to this negative feedback.

Feedback:
{feedback}
""",
    input_variables=["feedback"],
)

# ---------------- Branch ---------------- #

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "Positive",
        prompt2 | model1 | parser,
    ),
    (
        lambda x: x["sentiment"].sentiment == "Negative",
        prompt3 | model1 | parser,
    ),
    RunnableLambda(lambda x: "Could not determine sentiment."),
)

# ---------------- Final Chain ---------------- #

chain = (
    RunnablePassthrough.assign(
        sentiment=classifier_chain
    )
    | branch_chain
)

print("\nPositive Feedback:\n")
print(chain.invoke({"feedback": "This is a wonderful phone."}))

print("\nNegative Feedback:\n")
print(chain.invoke({"feedback": "The battery life is terrible."}))