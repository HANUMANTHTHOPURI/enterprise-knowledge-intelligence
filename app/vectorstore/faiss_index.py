import json
from pathlib import Path

import faiss
import numpy as np

from app.chunking.models import DocumentChunk


class FAISSVectorIndex:
    """Exact FAISS index for normalized enterprise chunk embeddings."""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero"
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""

        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ) -> None:
        """Add chunk embeddings to the FAISS index."""

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match index dimension"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count"
            )

        if not chunks:
            raise ValueError(
                "At least one chunk is required"
            )

        embeddings = np.ascontiguousarray(
            embeddings
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the highest-scoring chunks for a query vector."""

        if self.size == 0:
            raise ValueError(
                "Cannot search an empty FAISS index"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                "Query embedding must be a 1D vector"
            )

        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                "Query embedding dimension does not match index dimension"
            )

        query_matrix = np.ascontiguousarray(
            query_embedding.reshape(1, -1)
        )

        result_count = min(
            top_k,
            self.size,
        )

        scores, indices = self.index.search(
            query_matrix,
            result_count,
        )

        return [
            (
                self.chunks[index],
                float(score),
            )
            for index, score in zip(
                indices[0],
                scores[0],
            )
        ]

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """Persist the FAISS index and chunk metadata to disk."""

        if self.size == 0:
            raise ValueError(
                "Cannot save an empty FAISS index"
            )

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata = [
            chunk.model_dump(
                mode="json"
            )
            for chunk in self.chunks
        ]

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "FAISSVectorIndex":
        """Load a FAISS index and associated chunk metadata from disk."""

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        stored_chunks = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        chunks = [
            DocumentChunk.model_validate(item)
            for item in stored_chunks
        ]

        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS vector count does not match stored chunk metadata"
            )

        instance = cls(
            dimension=index.d,
        )

        instance.index = index
        instance.chunks = chunks

        return instance