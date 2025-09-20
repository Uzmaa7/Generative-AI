from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4o', temperature=0)

result = model.invoke("suggest me 5 indian names")

print(result)