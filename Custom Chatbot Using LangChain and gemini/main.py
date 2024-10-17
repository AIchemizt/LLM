import streamlit as st
from langchain_helper import setup_vector_db, create_retrieval_chain

st.title("Google GenerativeAI Demo QA 🌱")
btn = st.button("Create Knowledge Base")
if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = create_retrieval_chain()
    response = chain(question)

    st.header("Answer:")
    st.write(response["result"])