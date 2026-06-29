from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()

st.header('Research Tool')
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
user_input = st.text_input('Enter your prompt')

paper_input = st.selectbox('Select Research PaperName',['Attention is All you Need','BERT: Pre-Training of Deep Bidirectional Transformers','GPT-3:Language Models are Few-Shot Learners','Diffusion Models Beat GANs on Image Synthesis'])

style_input = st.selectbox('Select Explanation Style ',['Begineer-friendly','Technical','Code-Oriented','Mathematical'])

length_input = st.selectbox('Select Explanaction length',['Short (1-2 paragraphs)','Medium (3-5 paragraphs)','Long (detailed explanation)'])

template = PromptTemplate(
    template="""
    Please Summarize the reserach paper titled "{paper_input}" with the following specifications :
    Explanation Style:{style_input}
    Explanation Length:{length_input}
    1. Mathematical Details:
      - Include relevant mathematical equations if present in the paper.
      - Explain the mathematical concepts using simple , intuitive code snippets where applicable.
    2. Analogies:
      - Use relatable analogies to simplify complex table.
    If certain information is not available in the paper , respond with 'Insufficent Information available' instead of guessing.
    Ensure the summary us clear , accurate , and aligned with the provided style and length.

""",
input_variables= ['paper_input','style_input','length_input']

)

# fill the placeholder
prompt = template.invoke({
    'paper_input' : paper_input,
    'style_input':style_input,
    'length_input':length_input
})


if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

