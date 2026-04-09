"""
contrarag package
Core modules for ContraRAG — contradiction detection across multiple sources.
"""

from .ingestion import ingest_source
from .embedder import Embedder, get_embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .detector import ContradictionDetector
from .prompts import build_contradiction_prompt

__all__ = [
    "ingest_source",
    "Embedder",
    "get_embedder",
    "VectorStore",
    "Retriever",
    "ContradictionDetector",
    "build_contradiction_prompt",
]
