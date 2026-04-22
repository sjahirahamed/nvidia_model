import streamlit as st
import os
import time
import re

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

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# LangSmith (optional but recommended)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-demo"

# -------------------------------
# VECTOR EMBEDDING FUNCTION
# -------------------------------
def vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1"
        )

        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        final_documents = text_splitter.split_documents(docs[:30])

        vectors = FAISS.from_documents(
            final_documents,
            st.session_state.embeddings
        )

        st.session_state.vectors = vectors
        st.session_state.docs = final_documents


# -------------------------------
# SIMPLE RELEVANCE FUNCTION
# -------------------------------
def keyword_overlap(query, text):
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))

    if len(query_words) == 0:
        return 0

    return len(query_words & text_words) / len(query_words)


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="RAG App", layout="wide")

st.title("🚀 NVIDIA NIM RAG + LangSmith + Metrics")

llm = ChatNVIDIA(
    model="upstage/solar-10.7b-instruct"
)

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based only on the provided context.

<context>
{context}
</context>

Question: {input}
"""
)

user_question = st.text_input("💬 Ask a question from your documents")

if st.button("📚 Create Vector DB"):
    vector_embedding()
    st.success("Vector database created successfully ✅")


# -------------------------------
# MAIN QA + METRICS
# -------------------------------
if user_question:

    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the vector DB first")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        # ⏱ Start timer
        start_time = time.time()

        response = retrieval_chain.invoke(
            {"input": user_question},
            config={"run_name": "RAG_QA_Chain"}
        )

        end_time = time.time()
        latency = end_time - start_time

        answer = response["answer"]
        docs = response["context"]

        # -------------------------------
        # 📊 METRICS CALCULATION
        # -------------------------------

        num_chunks = len(docs)

        avg_chunk_len = (
            sum(len(doc.page_content) for doc in docs) / num_chunks
            if num_chunks > 0 else 0
        )

        relevance_scores = [
            keyword_overlap(user_question, doc.page_content)
            for doc in docs
        ]

        avg_relevance = (
            sum(relevance_scores) / len(relevance_scores)
            if relevance_scores else 0
        )

        total_text = user_question + answer + " ".join(
            doc.page_content for doc in docs
        )

        approx_tokens = len(total_text) / 4

        throughput = 1 / latency if latency > 0 else 0

        # -------------------------------
        # 🎯 UI OUTPUT
        # -------------------------------

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("⚡ Performance Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("⏱ Latency (sec)", f"{latency:.2f}")
        col2.metric("📄 Retrieved Chunks", num_chunks)
        col3.metric("📏 Avg Chunk Length", f"{avg_chunk_len:.0f}")

        col1.metric("🎯 Relevance Score", f"{avg_relevance:.2f}")
        col2.metric("🧠 Approx Tokens", f"{approx_tokens:.0f}")
        col3.metric("🚀 Throughput (Q/s)", f"{throughput:.2f}")

        # -------------------------------
        # 🔍 CONTEXT DISPLAY
        # -------------------------------
        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(docs):
                score = relevance_scores[i] if i < len(relevance_scores) else 0
                st.write(f"Chunk {i+1} | Score: {score:.2f}")
                st.write(doc.page_content)
                st.write("------")
