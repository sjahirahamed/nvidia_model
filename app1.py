import streamlit as st
import os
import time

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# -------------------------------
# LOAD ENV VARIABLES
# -------------------------------
load_dotenv()

# NVIDIA API KEY
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# 🔥 LANGSMITH SETUP
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-demo"

# -------------------------------
# VECTOR EMBEDDING FUNCTION
# -------------------------------
def vector_embedding():
    if "vectors" not in st.session_state:

        # Embedding model
        st.session_state.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1"
        )

        # Load documents
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()

        # Split documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs[:30]
            )
        )

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🚀 NVIDIA NIM RAG + LangSmith")

# ✅ FIXED MODEL (non-deprecated)
llm = ChatNVIDIA(
    model="upstage/solar-10.7b-instruct"
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based only on the provided context.

<context>
{context}
</context>

Question: {input}
"""
)

# User input
user_question = st.text_input("Ask a question from your documents")

# Button to create vector DB
if st.button("Create Vector DB"):
    vector_embedding()
    st.success("Vector database is ready ✅")

# -------------------------------
# QUESTION ANSWERING
# -------------------------------
if user_question:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Create Vector DB' first")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        start = time.process_time()

        # 🔥 LangSmith tracking with run name
        response = retrieval_chain.invoke(
            {"input": user_question},
            config={"run_name": "RAG_QA_Chain"}
        )

        st.write("### ✅ Answer:")
        st.write(response["answer"])

        st.write(f"⏱ Response time: {time.process_time() - start:.2f} sec")

        # Show retrieved chunks
        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content)
                st.write("------")