"""Assistant LLM/RAG spécialisé DevOps & MLOps.

Pipeline : ingestion (PDF/MD/txt) → embeddings (SentenceTransformers) →
index FAISS → génération (Transformers) avec prompt pédagogique.
"""
from .config import Settings, settings
from .embed import Embedder
from .index import SearchHit, VectorStore
from .ingest import Chunk, build_corpus
from .llm import LLM
from .pipeline import RAGAnswer, RAGPipeline
from .summarize import summarize_document

__all__ = [
    "Settings",
    "settings",
    "Embedder",
    "VectorStore",
    "SearchHit",
    "Chunk",
    "build_corpus",
    "LLM",
    "RAGPipeline",
    "RAGAnswer",
    "summarize_document",
]
