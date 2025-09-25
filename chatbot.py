from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id =  "openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

st.set_page_config(
    page_title="Ask Me Anything!",
    layout="centered"
)


st.title("Ask Me Anything!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role" : "system" , "content" : "You are a helpful assistant."}]

for message in st.session_state.chat_history:

    if(message['role'] != 'system'):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown("How can i help you today!")


user_input = st.chat_input("How can i help you....")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.chat_history.append({"role" : "user", "content" : user_input})

    lc_chat_history = []

    for message in st.session_state.chat_history:
        if(message["role"] == "user"):
            lc_chat_history.append(HumanMessage(content = message["content"]))
        
        elif(message["role"] == "assistant"):
            lc_chat_history.append(AIMessage(content = message["content"]))
        
        elif (message["role"] == "system"):
            lc_chat_history.append(SystemMessage(content= message["content"]))
    


    result = model.invoke(lc_chat_history)
    response = result.content

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({"role" : "assistant", "content" : response})


    
