"""
Central configuration for the RAG Q&A system.
All tunable parameters are pulled from environment variables (see .env.example)
so the pipeline can be reconfigured without touching code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))

# Where FAISS indexes get persisted to disk (one sub-folder per uploaded doc set)
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore_index")

if not ANTHROPIC_API_KEY:
    # Don't crash on import (e.g. during tests) — just warn. app.py checks this too.
    print("[config] WARNING: ANTHROPIC_API_KEY is not set. Set it in your .env file.")
  
