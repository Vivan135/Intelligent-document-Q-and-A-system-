"""
Vector store layer: embeds chunks and builds/loads a FAISS index.
Uses a local sentence-transformers model for embeddings (no extra API
cost/latency) — swap EMBEDDING_MODEL in .env if you want a different one.
"""
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vectorstore(chunks: List[Document], index_name: str) -> FAISS:
    """Embed chunks and build a fresh FAISS index, then persist to disk."""
    store = FAISS.from_documents(chunks, get_embeddings())
    path = os.path.join(VECTORSTORE_DIR, index_name)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    store.save_local(path)
    return store


def load_vectorstore(index_name: str) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    path = os.path.join(VECTORSTORE_DIR, index_name)
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )


def index_exists(index_name: str) -> bool:
    path = os.path.join(VECTORSTORE_DIR, index_name)
    return os.path.exists(os.path.join(path, "index.faiss"))
