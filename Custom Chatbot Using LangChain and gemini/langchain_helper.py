import os
import google.generativeai as genai
from langchain.document_loaders.csv_loader import CSVLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema.runnable import Runnable
from typing import Any, List, Optional, Dict
from pydantic import Field
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()       

# Set up the API key using the environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration
CSV_FILE_PATH = "pi_faqs.csv"
INSTRUCTOR_MODEL_NAME = 'hkunlp/instructor-large'

# Setup Google Generative AI
def setup_genai():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# Load CSV data
def load_csv_data(file_path, source_column):
    loader = CSVLoader(file_path=file_path, source_column=source_column)
    return loader.load()

# Custom Embeddings class
class InstructorEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.instructor = INSTRUCTOR(model_name)
    
    def embed_documents(self, texts):
        return self.instructor.encode(texts)
    
    def embed_query(self, text):
        return self.instructor.encode([text])[0]

# Google Generative AI Wrapper
class GoogleGenerativeAIWrapper(LLM, Runnable):
    model: Any = Field(..., description="Google GenerativeAI model")

    def __init__(self, model: Any):
        super().__init__(model=model)

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.model.generate_content(prompt)
        return response.text
    
    @property    
    def _identifying_params(self):
        return {"model_name": self.model.model_name}
    
    @property
    def _llm_type(self):
        return "google_generative_ai"
    
    def invoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        return self._call(input["prompt"])

# Setup vector database
def setup_vector_db():
    data = load_csv_data(CSV_FILE_PATH, "prompt")
    instructor_embeddings = InstructorEmbeddings(INSTRUCTOR_MODEL_NAME)
    return FAISS.from_documents(documents=data, embedding=instructor_embeddings)

# Create retrieval chain
def create_retrieval_chain():

    # Setup Google Generative AI
    llm = setup_genai()
    llm_wrapper = GoogleGenerativeAIWrapper(model=llm)

    # Load CSV data
    data = load_csv_data(CSV_FILE_PATH, "prompt")

    # Setup embeddings and vector database
    instructor_embeddings = InstructorEmbeddings(INSTRUCTOR_MODEL_NAME)
    vectordb = setup_vector_db()
    retriever = vectordb.as_retriever()
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question} """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    return RetrievalQA.from_chain_type(
        llm=llm_wrapper,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )