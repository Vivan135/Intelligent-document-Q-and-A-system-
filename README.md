# Intelligent-document-Q-and-A-system-
Built a Retrieval-Augmented Generation (RAG) system using Claude to answer questions from large documents (PDFs, research papers, company reports). The system retrieves relevant chunks using vector embeddings and generates accurate, context-aware answers. Key Features: PDF ingestion + chunking pipeline Vector database pipeline 
# Intelligent Document Q&A System (Claude-powered RAG)

A Retrieval-Augmented Generation system that answers questions from large
documents (PDFs, research papers, company reports) using Claude, with
cited sources so answers can be verified.

## Architecture

```
PDF upload
   -> ingest.py      (load pages, chunk with overlap)
   -> vectorstore.py  (embed chunks, build/persist FAISS index)
   -> qa_chain.py     (embed question, retrieve top-k chunks, ask Claude)
   -> app.py          (Streamlit UI: upload, index, ask, view sources)
```

## Setup

```bash
git clone <your-repo-url>
cd rag-qa-system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more PDFs in the sidebar.
2. Click **Build / rebuild index** (embeds + saves a FAISS index to
   `vectorstore_index/<name>/`).
3. Ask a question — Claude answers using only the retrieved chunks and
   cites `[Source N]` for each fact, shown in an expandable panel below
   the answer.

Re-open the app later and click **Load existing index** to reuse a
previously built index without re-embedding.

## Key design choices

- **Chunking**: `RecursiveCharacterTextSplitter` with overlap (default
  1000 chars, 150 overlap) so facts near chunk boundaries aren't lost.
- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`
  by default) — no extra API cost or latency for embedding, only
  generation calls Claude.
- **Retrieval**: FAISS similarity search, top-k configurable via `.env`.
- **Grounding**: the system prompt instructs Claude to answer only from
  the retrieved context and to say when the answer isn't present, to
  reduce hallucination.
- **Citations**: each retrieved chunk carries its source filename and
  page number in metadata, surfaced both in Claude's `[Source N]`
  references and in the UI's expandable source panel.

## Extending this project

- Swap FAISS for **Pinecone** for a hosted/scalable vector store (see
  `vectorstore.py` — the interface is the same `similarity_search` call).
- Add re-ranking (e.g. Cohere rerank) after initial retrieval for higher
  precision on large corpora.
- Add conversation memory so follow-up questions can reference prior
  turns.
- Add evaluation: a small set of Q&A pairs with expected sources to
  track retrieval precision/recall as you tune chunk size and top-k.

## Tech stack

Python, LangChain, FAISS, Streamlit, Claude API (Anthropic SDK)
