# Import necessary libraries and modules
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Loader for PDF documents
# from langchain_community.document_loaders import PDFPlumberLoader
from langchain import hub  # Hub for model retrieval
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings  # Embeddings for OpenAI models
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitter for document chunking
from langchain.vectorstores.chroma import Chroma  # Chroma vector store for document vectors
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser


# Load environment variables from .env file
class ChatWithDocs:

    def __init__(self, API_KEY, filepath):

        # Initialize PyPDFDirectoryLoader to load PDF documents from a directory
        loader = PyPDFDirectoryLoader(filepath)

        # Load document data using the loader
        data = loader.load()

        # Initialize RecursiveCharacterTextSplitter for text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Split documents into chunks using the text splitter
        splits = text_splitter.split_documents(data)

        # Initialize Chroma vector store with OpenAI embeddings from the document splits
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))

        # Create a retriever from the vector store
        self.retriever = vectorstore.as_retriever()

        # Load pre-defined prompt model from the hub
        prompt = hub.pull("rlm/rag-prompt")

        #  Initialize ChatOpenAI model for language generation using OpenAI's GPT-3.5-turbo
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)
        # Define the processing chain for question answering
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
)

    def format_docs(self, data):
        """
        Format document data into a string with page content separated by newlines.

        Paramaters:
        - data: List of document objects.

        Returns:
        - str: Formatted document string.
        """
        return "\n\n".join(doc.page_content for doc in data)
    
    def questions(self, question):
        return(self.rag_chain.invoke(question))

    def done(self):
        self.vectorstore.delete_collection()