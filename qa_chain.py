"""
Core RAG logic: retrieve relevant chunks from FAISS, build a grounded
prompt, call Claude for the answer, and return both the answer and the
source chunks so the UI can display citations.
"""
from dataclasses import dataclass
from typing import List
import anthropic
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K

SYSTEM_PROMPT = """You are a precise document Q&A assistant. Answer the user's \
question using ONLY the provided context excerpts. If the answer isn't in the \
context, say so clearly instead of guessing. When you use a fact, reference \
which excerpt it came from using [Source N] notation."""


@dataclass
class QAResult:
    answer: str
    sources: List[Document]


def _format_context(chunks: List[Document]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        blocks.append(
            f"[Source {i}] (file: {src}, page: {page})\n{chunk.page_content}"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, vectorstore: FAISS, top_k: int = TOP_K) -> QAResult:
    """Retrieve relevant chunks and generate a cited answer via Claude."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    retrieved_chunks = vectorstore.similarity_search(question, k=top_k)
    context = _format_context(retrieved_chunks)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context excerpts:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the context above, citing [Source N] "
                    "for each fact you use."
                ),
            }
        ],
    )

    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return QAResult(answer=answer_text, sources=retrieved_chunks)
