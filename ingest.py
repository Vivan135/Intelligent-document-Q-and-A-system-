"""
Ingestion pipeline: load PDFs, split into overlapping chunks, and attach
metadata (source filename + page number) so answers can cite where they
came from.
"""
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF into one LangChain Document per page."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level documents into smaller overlapping chunks.
    Overlap preserves context across chunk boundaries so answers
    don't lose meaning when a fact spans a split point.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Tag each chunk with a stable id for citation display
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_pdfs(file_paths: List[str]) -> List[Document]:
    """Full ingestion pipeline for a batch of PDFs -> ready-to-embed chunks."""
    all_docs: List[Document] = []
    for path in file_paths:
        pages = load_pdf(path)
        all_docs.extend(pages)
    return chunk_documents(all_docs)
