"""
vector_store.py
Manages FAISS vector storage with one index per source document.
"""

import numpy as np
import faiss
from typing import Optional


class VectorStore:
    """
    Wraps FAISS with one index per source document.
    """

    def __init__(self) -> None:
        self.indices: dict = {}
        self.documents: dict = {}

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        grouped_chunks: dict = {}
        grouped_embeddings: dict = {}

        for chunk, embedding in zip(chunks, embeddings):
            sid = chunk["source_id"]
            if sid not in grouped_chunks:
                grouped_chunks[sid] = []
                grouped_embeddings[sid] = []
            grouped_chunks[sid].append(chunk)
            grouped_embeddings[sid].append(embedding)

        for source_id, items in grouped_chunks.items():
            vecs = np.array(grouped_embeddings[source_id], dtype=np.float32)
            faiss.normalize_L2(vecs)
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            self.indices[source_id] = index
            self.documents[source_id] = items

    def query_source(
        self,
        source_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        if source_id not in self.indices:
            return []
        index = self.indices[source_id]
        docs = self.documents[source_id]
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        k = min(top_k, len(docs))
        distances, indices = index.search(query_vec, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(docs):
                results.append({
                    "text": docs[idx]["text"],
                    "source_id": source_id,
                    "chunk_index": docs[idx]["chunk_index"],
                    "distance": float(distances[0][i]),
                })
        return results

    def list_sources(self) -> list[str]:
        return list(self.indices.keys())

    def clear_all(self) -> None:
        self.indices = {}
        self.documents = {}
