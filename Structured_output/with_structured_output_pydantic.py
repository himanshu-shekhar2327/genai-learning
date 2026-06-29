from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal
from pydantic import BaseModel, Field, Optio

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

# Schema 
# class Review(TypedDict):
#     summary : str
#     sentiment : str

class Review(TypedDict):
    summary : str = Field(description='A brief summary of the review')
    sentiment : Literal['pos','neg'] = Field(description='Return sentiment of the review')






strucutred_model = model.with_structured_output(Review)

result = strucutred_model.invoke("""
The hardware is great , but the software feels bloated . There are are too many pre-intalled apps that i can't remove. also , the UI looks outdated compare to other brands . Hoping for a software updatee to fix this .
""")

print(result)
print(dict(result))