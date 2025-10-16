from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on  {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarise the following text {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

finali_chain = RunnableSequence(report_generation_chain, branch_chain)

print(finali_chain.invoke({'topic': 'cricket'})) 