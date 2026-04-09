"""
retriever.py
Retrieves relevant chunks from each source document for a given query.
"""

from .vector_store import VectorStore
from .embedder import Embedder


class Retriever:
    """
    Retrieves the most relevant chunks from every loaded source for a query.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        """
        Initialize the retriever with a vector store and embedder.

        Args:
            vector_store: The VectorStore instance holding all source collections.
            embedder: The Embedder instance used to embed the query.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_all_sources(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, list[str]]:
        """
        Embed the query and retrieve the top-k chunks from every source.

        Args:
            query: The user's research question.
            top_k: Number of chunks to retrieve per source.

        Returns:
            A dict mapping source_id to a list of relevant chunk text strings.
            Sources that return no results are excluded.
        """
        query_embedding = self.embedder.embed([query])[0]
        source_ids = self.vector_store.list_sources()

        results = {}
        for source_id in source_ids:
            chunks = self.vector_store.query_source(
                source_id=source_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            if chunks:
                results[source_id] = [chunk["text"] for chunk in chunks]

        return results
