from dotenv import load_dotenv  # Load environment variables
import os  # Operating system module
from tkinter import Tk 
from tkinter.filedialog import askdirectory
import Son_of_Son_of_Anton
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

Tk().withdraw()
filename = askdirectory(title="Select a folder")
chat = Son_of_Son_of_Anton.ChatWithDocs(API_KEY, filename)

# # Continue asking for user input until an empty question is entered
while True:
    # Prompt the user for a question
    question = input("Question: ")

    # If the entered question is empty, exit the loop
    if not question:
        break
  # Ask question method
    answer = chat.questions(question)
    print(answer)

chat.done()
# loader = PyPDFDirectoryLoader(filename)

# # Load document data using the loader
# data = loader.load()

# # Initialize RecursiveCharacterTextSplitter for text splitting
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # Split documents into chunks using the text splitter
# splits = text_splitter.split_documents(data)

# # Initialize Chroma vector store with OpenAI embeddings from the document splits
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=API_KEY))

# # Create a retriever from the vector store
# retriever = vectorstore.as_retriever()

# # Load pre-defined prompt model from the hub
# prompt = hub.pull("rlm/rag-prompt")

# # Initialize ChatOpenAI model for language generation using OpenAI's GPT-3.5-turbo
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)

# def format_docs(data):
#     """
#     Format document data into a string with page content separated by newlines.

#     Paramaters:
#     - data: List of document objects.

#     Returns:
#     - str: Formatted document string.
#     """
#     return "\n\n".join(doc.page_content for doc in data)

# # Define the processing chain for question answering
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Define a list of sample questions for the company's financial data
# #questions = [
#     #"What is the name of the company?",
#     #"What are the products the company makes?",
#     #"What are the net sales?",
#     #"What was the average stock price?"
# #]

# # Print a prompt to ask the user for a question and exit instructions
# print("Ask a question (press enter to exit):")


# # Iterate through each question, invoke the processing chain, and print the results
# #for question in questions:
#     #print(rag_chain.invoke(question))

# # Delete the vector store collection after processing
# vectorstore.delete_collection()