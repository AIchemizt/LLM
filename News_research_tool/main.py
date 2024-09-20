# Import the SentenceTransformer class for text embedding
from sentence_transformers import SentenceTransformer

# Import the os module for operating system dependent functionality
import os

# Import Streamlit for creating web applications
import streamlit as st

# Import pickle for object serialization
import pickle

# Import time module for time-related functions
import time

# Import RetrievalQAWithSourcesChain for question answering with source retrieval
from langchain.chains import RetrievalQAWithSourcesChain

# Import RecursiveCharacterTextSplitter for splitting text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import UnstructuredURLLoader for loading content from URLs
from langchain.document_loaders import UnstructuredURLLoader

# Import FAISS for efficient similarity search and clustering of dense vectors
from langchain.vectorstores import FAISS

# Import ChatGroq, a language model API
from langchain_groq import ChatGroq

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Import numpy for numerical operations
import numpy as np

# Import the base Embeddings class for creating custom embeddings
from langchain.embeddings.base import Embeddings

# Import the Document class for representing text documents
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Initialize the SentenceTransformer model for text embedding
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Define the file path for storing the vector index
file_path = "vector_index.pkl"

# Set the title of the Streamlit app
st.title("Volk Research Tool")

# Set the title of the sidebar in the Streamlit app
st.sidebar.title("Your Article URLs")

# Initialize an empty list to store URLs
urls = []

# Create three text input fields for URLs in the sidebar
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Create a button to process the entered URLs
process_url_clicked = st.sidebar.button("Process URLs")

# Create an empty placeholder in the main area of the app
main_placefolder = st.empty()

# Initialize the ChatGroq language model with API key and model name
llm = ChatGroq(
    groq_api_key="your_api_key",  # Replace with your actual API key
    model_name="llama-3.1-70b-versatile",
)

# Define a custom Embeddings class
class CustomEmbeddings(Embeddings):
    # Initialize the CustomEmbeddings with pre-computed embeddings
    def __init__(self, embeddings):
        self.embeddings = embeddings

    # Method to return pre-computed embeddings for documents
    def embed_documents(self, texts):
        return self.embeddings

    # Method to return the first pre-computed embedding for queries
    def embed_query(self, text):
        return self.embeddings[0]

# Check if the "Process URLs" button was clicked
if process_url_clicked:
    # Create a loader object to fetch content from the provided URLs
    loader = UnstructuredURLLoader(urls=urls)
    # Display a message indicating that data loading has started
    main_placefolder.text("Data Loading...Started...")
    # Load the data from the URLs
    data = loader.load()

    # Create a text splitter object to break text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_overlap=200,
    )
    # Display a message indicating that text splitting has started
    main_placefolder.text("Text Splitter...Started...")
    # Split the loaded documents into smaller chunks
    docs = text_splitter.split_documents(data)

    # Initialize an empty list to store embeddings
    embeddings = []
    # Loop through each document and create its embedding
    for doc in docs:
        embedding = model.encode(doc.page_content)
        embeddings.append(embedding)

    # Convert the list of embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Create a list of Document objects from the split documents
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]

    # Create a CustomEmbeddings object with the computed embeddings
    custom_embeddings = CustomEmbeddings(embeddings_array.tolist())

    # Create a FAISS vector store from the documents and custom embeddings
    vectorstore = FAISS.from_documents(
        documents,
        embedding=custom_embeddings
    )
    # Display a message indicating that embedding vector building has started
    main_placefolder.text("Embedding Vector Started Building...Started...")

    # Print a success message for FAISS index creation
    print("FAISS index created successfully")

    # Save the FAISS index locally
    vectorstore.save_local("my_faiss_index")
    # Print a success message for saving the FAISS index
    print("FAISS index saved locally")

    # Open a file to save the vector store
    with open(file_path, "wb") as f:
        # Serialize and save the vector store to the file
        pickle.dump(vectorstore, f)

# Create a text input field for the user's question
query = main_placefolder.text_input("Question: ")
# Check if a query has been entered
if query:
    # Check if the vector index file exists
    if os.path.exists(file_path):
        # Open the vector index file
        with open(file_path, "rb") as f:
            # Load the vector index from the file
            vectorIndex = pickle.load(f)
            # Create a retrieval QA chain using the loaded vector index
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
            # Get the answer to the query using the QA chain
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display the "Answer" header
            st.header("Answer")
            # Display the answer to the query
            st.subheader(result["answer"])

            # Get the sources for the answer, if available
            sources = result.get("sources", "")
            # Check if there are any sources
            if sources:
                # Display the "Sources" subheader
                st.subheader("Sources:")
                # Split the sources string into a list
                sources_list = sources.split("\n")
                # Display each source
                for source in sources_list:
                    st.write(source)