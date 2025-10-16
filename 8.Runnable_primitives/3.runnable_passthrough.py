from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Write the explanation of following joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()



joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)
})

finali_chain = RunnableSequence(joke_generator_chain, parallel_chain)

print(finali_chain.invoke({'topic': 'cricket'}))