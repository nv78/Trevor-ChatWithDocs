# Import necessary libraries and modules
import openai  # OpenAI language model API
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Loader for PDF documents
# from langchain_community.document_loaders import PDFPlumberLoader
import bs4
from langchain import hub  # Hub for model retrieval
from langchain_community.chat_models import ChatOpenAI  # Chat model for language generation
from langchain_openai import OpenAIEmbeddings  # Embeddings for OpenAI models
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitter for document chunking
from langchain.vectorstores.chroma import Chroma  # Chroma vector store for document vectors
from dotenv import load_dotenv  # Load environment variables
import os  # Operating system module

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize PyPDFDirectoryLoader to load PDF documents from a directory
loader = PyPDFDirectoryLoader("data/10-Ks copy/")

# Load document data using the loader
data = loader.load()

# Initialize RecursiveCharacterTextSplitter for text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split documents into chunks using the text splitter
splits = text_splitter.split_documents(data)

# Initialize Chroma vector store with OpenAI embeddings from the document splits
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Load pre-defined prompt model from the hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize ChatOpenAI model for language generation using OpenAI's GPT-3.5-turbo
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)

def format_docs(data):
    """
    Format document data into a string with page content separated by newlines.

    Paramaters:
    - data (list): List of document objects.

    Returns:
    - str: Formatted document string.
    """
    return "\n\n".join(doc.page_content for doc in data)

# Define the processing chain for question answering
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define a list of sample questions for the company's financial data
questions = [
    "What is the name of the company?",
    "What are the products the company makes?",
    "What are the net sales?",
    "What was the average stock price?"
]

# Iterate through each question, invoke the processing chain, and print the results
for question in questions:
    print(rag_chain.invoke(question))

# Delete the vector store collection after processing
vectorstore.delete_collection()