from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#static
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id =  "openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)