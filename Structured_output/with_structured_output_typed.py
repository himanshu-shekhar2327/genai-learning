from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

# Schema 
# class Review(TypedDict):
#     summary : str
#     sentiment : str

class Review(TypedDict):
    summary : Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, 'Return sentiment of the review either negative , positive, neutral']





strucutred_model = model.with_structured_output(Review)

result = strucutred_model.invoke("""
The hardware is great , but the software feels bloated . There are are too many pre-intalled apps that i can't remove. also , the UI looks outdated compare to other brands . Hoping for a software updatee to fix this .
""")

print(result)
print(result['summary'])
print(result['sentiment'])