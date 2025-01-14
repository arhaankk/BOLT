from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(  
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
pinecone_api_key = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = pinecone_api_key

model = ChatOpenAI(
    model="gpt-4o"
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

data = [
    "data/transfer_credits_a_level.txt",
    "data/transfer_credits_AP.txt",
    "data/transfer_credits_IB.txt",
    "data/academic_regulation.txt",
    "data/major_minor.txt",
    "data/course_withdrawal_winter.txt",
    "data/course_withdrawal_summer.txt",
    "data/graduation.txt"
]

docs = []
for file in data:
    loader = TextLoader(file)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

index_name = "bolt-chatbot"
vectorstore = PineconeVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    index_name=index_name
)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})
parser = StrOutputParser()

system_prompt = (
    """
    Answer the question based on the context below. You are an academic advisor for University of British Columbia (UBCO) students. 
    Don't start by saying, based on provided context. Only answer relevant question, if you don't know the answer to something, reply I can only help you with academic advising.
    If the question is course-related, say that please refer to the course advising chatbot.
    Every time you use use provided data, I need you to cite the source of data. Example, Source: https://you.ubc.ca/applying-ubc/applied/first-year-credit/. 
    Do not say, Source: provided context, a source is always a URL. Always add source on a new line.
    """

    "Context: {context}"

)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)

chat_history = []


class UserQuestion(BaseModel):
    question: str

    class Config:
        from_attributes = True


@app.get("/", status_code=200)
async def root():
    return {"message": "Welcome to Bolt"}

@app.post("/ask/", status_code=200)
async def ask_question(user_question: UserQuestion):
    question = user_question.question
    response = rag_chain.invoke(
        {"input": question, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=response["answer"]),
        ]
    )
    return JSONResponse(content={"question": question, "answer": response["answer"]})