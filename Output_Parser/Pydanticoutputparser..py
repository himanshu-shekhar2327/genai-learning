from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation',

)

model = ChatHuggingFace(llm=llm)

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
Generate the name, age and city of a fictional {place} person.

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Create the prompt
# prompt = template.invoke({"place": "Indian"})

# # Send it to the model
# result = model.invoke(prompt)

# # Parse the response
# person = parser.parse(result.content)

# print(person)
# print(person.name)
# print(person.age)
# print(person.city)

# Using chain  

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)