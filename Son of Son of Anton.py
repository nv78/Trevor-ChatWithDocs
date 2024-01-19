import openai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PDFPlumberLoader
# from langchain.document_loaders import PDFPlumberLoader
import bs4
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
loader = PyPDFDirectoryLoader("data/10-Ks copy/")

data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)


def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "What are the names of the companies?",
    "What are the products the companies make?",
    "What are the net sales of each company?",
    "What was the average stock price of each company?"
]

for question in questions:
  print(rag_chain.invoke(question))
vectorstore.delete_collection()