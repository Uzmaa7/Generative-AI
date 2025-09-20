from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 300)

documents = [

    "Virat Kohli, known as King Kohli, is a modern Indian batting maestro celebrated for his consistency and aggressive captaincy."
    "Sachin Tendulkar, widely regarded as the Little Master,is an Indian legend with the most runs in both Test and ODI cricket history."
    "Sir Donald Bradman, an Australian icon, holds the record for the highest-ever Test batting average of 99.94, a feat considered nearly impossible to achieve."
    "Shane Warne, a legendary Australian leg-spinner, revolutionized the art of spin bowling and is one of the leading wicket-takers in Test cricket."

]

query = "tell me about virat kohli"

doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]


print(query)

print(documents[index])

print("similarity_score_is:", score)

