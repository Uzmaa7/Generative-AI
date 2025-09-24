from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# static prompts

llm = HuggingFaceEndpoint(
    repo_id =  "openai/gpt-oss-20b",
    task = "text-generation"
)


model = ChatHuggingFace(llm = llm)

st.header('Research Tool')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write (result.content)



