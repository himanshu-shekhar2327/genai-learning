from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']

)

prompt2 = PromptTemplate(
    template = 'Explan the following joke - {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt , model , parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model, parser)
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))




