"""
vector_store.py
Manages ChromaDB vector storage with one collection per source document.
"""

import chromadb
import re
from typing import Optional


class VectorStore:
    """
    Wraps ChromaDB with one collection per source document.
    """

    def __init__(self) -> None:
        """
        Initialize an in-memory ChromaDB client.
        """
        self.client = chromadb.EphemeralClient()
        self.collections: dict = {}

    def _sanitize_name(self, source_id: str) -> str:
        """
        Sanitize a source_id to be a valid ChromaDB collection name.

        Args:
            source_id: Raw source identifier string.

        Returns:
            A sanitized string with only alphanumerics and underscores.
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", source_id)

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add chunks and their embeddings to the appropriate per-source collection.

        Args:
            chunks: List of chunk dicts with keys: text, source_id, chunk_index.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        grouped: dict = {}
        for chunk, embedding in zip(chunks, embeddings):
            sid = chunk["source_id"]
            if sid not in grouped:
                grouped[sid] = []
            grouped[sid].append((chunk, embedding))

        for source_id, items in grouped.items():
            name = self._sanitize_name(source_id)
            if source_id not in self.collections:
                collection = self.client.create_collection(name=name)
                self.collections[source_id] = collection
            else:
                collection = self.collections[source_id]

            collection.add(
                embeddings=[item[1] for item in items],
                documents=[item[0]["text"] for item in items],
                metadatas=[
                    {
                        "source_id": item[0]["source_id"],
                        "chunk_index": item[0]["chunk_index"],
                    }
                    for item in items
                ],
                ids=[
                    f"{item[0]['source_id']}_{item[0]['chunk_index']}"
                    for item in items
                ],
            )

    def query_source(
        self,
        source_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Query a single source's collection for the most relevant chunks.

        Args:
            source_id: The source to query.
            query_embedding: The embedded query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, source_id, chunk_index, distance.
        """
        if source_id not in self.collections:
            return []
        collection = self.collections[source_id]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        output = []
        for i, doc in enumerate(results["documents"][0]):
            output.append(
                {
                    "text": doc,
                    "source_id": source_id,
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "distance": results["distances"][0][i],
                }
            )
        return output

    def list_sources(self) -> list[str]:
        """
        Return all source_ids currently stored.

        Returns:
            List of source_id strings.
        """
        return list(self.collections.keys())

    def clear_all(self) -> None:
        """
        Delete all collections and reset internal state.
        """
        for source_id in list(self.collections.keys()):
            name = self._sanitize_name(source_id)
            self.client.delete_collection(name=name)
        self.collections = {}
