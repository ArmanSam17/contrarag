"""
embedder.py
Wraps sentence-transformers for local text embedding in ContraRAG.
"""

from sentence_transformers import SentenceTransformer
from typing import Optional


class Embedder:
    """
    Wraps a sentence-transformers model to produce text embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the sentence-transformers model.

        Args:
            model_name: The name of the sentence-transformers model to load.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]


_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """
    Return the module-level singleton Embedder instance.
    Creates it on first call.

    Returns:
        The shared Embedder instance.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
