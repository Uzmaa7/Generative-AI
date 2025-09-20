from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model ='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the Capital of India",
    "paris is the capital of france",
    "kolkata is the capital of west bengal"
]

result = model.embed_documents(documents)

print(str(result))