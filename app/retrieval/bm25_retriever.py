import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking.models import DocumentChunk
from app.retrieval.models import RetrievalResult


class BM25Retriever:
    """Retrieve enterprise chunks using BM25 lexical relevance."""

    TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks

        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]

        self.model = BM25Okapi(
            self.tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return chunks ranked by BM25 relevance."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_tokens = self._tokenize(query)

        scores = np.asarray(
            self.model.get_scores(query_tokens),
            dtype=np.float32,
        )

        result_count = min(
            top_k,
            len(self.chunks),
        )

        top_indices = np.argsort(scores)[::-1][
            :result_count
        ]

        return [
            RetrievalResult(
                rank=rank,
                score=float(scores[index]),
                chunk=self.chunks[index],
            )
            for rank, index in enumerate(
                top_indices,
                start=1,
            )
        ]

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> list[str]:
        """Convert text into normalized lexical tokens."""

        return [
            token.lower()
            for token in cls.TOKEN_PATTERN.findall(text)
        ]