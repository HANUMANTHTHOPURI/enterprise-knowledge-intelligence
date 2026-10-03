from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingEncoder:
    """Encode text into normalized dense vector representations."""

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        self.model_name = (
            model_name
            if model_name is not None
            else settings.embedding_model
        )

        self.model = SentenceTransformer(self.model_name)

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise ValueError(
                "Embedding model did not report an embedding dimension"
            )

        self.dimension = dimension

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Encode multiple texts into normalized embedding vectors."""

        if not texts:
            raise ValueError("At least one text is required")

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding text cannot be empty")

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode a single search query into a normalized vector."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )