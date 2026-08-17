"""
Streamlit front-end for the Intelligent Document Q&A System.
Upload PDFs -> they're chunked + embedded into FAISS -> ask questions ->
Claude answers using retrieved chunks, with sources shown for verification.
"""
import os
import tempfile
import streamlit as st

from src.config import ANTHROPIC_API_KEY
from src.ingest import ingest_pdfs
from src.vectorstore import build_vectorstore, load_vectorstore, index_exists
from src.qa_chain import answer_question

st.set_page_config(page_title="Document Q&A (RAG + Claude)", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Q&A System")
st.caption("Upload documents, then ask questions. Answers are grounded in your files, with sources cited.")

if not ANTHROPIC_API_KEY:
    st.warning("ANTHROPIC_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    index_name = st.text_input("Index name", value="default", help="Lets you keep separate knowledge bases.")

    if st.button("Build / rebuild index", disabled=not uploaded_files):
        with st.spinner("Ingesting, chunking, and embedding documents..."):
            tmp_paths = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    tmp_paths.append(path)
                chunks = ingest_pdfs(tmp_paths)
                st.session_state.vectorstore = build_vectorstore(chunks, index_name)
        st.success(f"Indexed {len(uploaded_files)} file(s) into '{index_name}'.")

    if st.button("Load existing index", disabled=not index_exists(index_name)):
        with st.spinner("Loading saved index..."):
            st.session_state.vectorstore = load_vectorstore(index_name)
        st.success(f"Loaded index '{index_name}'.")

st.header("2. Ask a question")
question = st.text_input("Your question about the uploaded documents")
ask = st.button("Ask", type="primary", disabled=st.session_state.vectorstore is None)

if ask and question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = answer_question(question, st.session_state.vectorstore)
            st.session_state.history.append((question, result))
        except Exception as e:
            st.error(f"Error: {e}")

for q, result in reversed(st.session_state.history):
    st.markdown(f"**Q: {q}**")
    st.write(result.answer)
    with st.expander(f"View {len(result.sources)} source excerpt(s)"):
        for i, chunk in enumerate(result.sources, start=1):
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            st.markdown(f"**[Source {i}]** `{os.path.basename(str(src))}`, page {page}")
            st.text(chunk.page_content[:500] + ("..." if len(chunk.page_content) > 500 else ""))
    st.divider()

if st.session_state.vectorstore is None:
    st.info("Upload PDFs and build an index in the sidebar to get started.")
