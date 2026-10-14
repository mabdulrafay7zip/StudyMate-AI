"""Retrieval for StudyMate AI.

Two retrievers behind one interface:

- ``EmbeddingRetriever`` — semantic search with sentence-transformers
  (``all-MiniLM-L6-v2``). Preferred when the package and model are available.
- ``TfidfRetriever`` — classic TF-IDF cosine similarity (scikit-learn).
  Always works, no downloads, and is the automatic fallback.

``build_retriever()`` tries embeddings first and silently falls back to
TF-IDF, so the app runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingest import Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BaseRetriever:
    name = "base"

    def fit(self, chunks: list[Chunk]) -> "BaseRetriever":
        raise NotImplementedError

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> "TfidfRetriever":
        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])
        return self

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in order]


class EmbeddingRetriever(BaseRetriever):
    name = "embeddings"

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._embeddings = None
        self._chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> "EmbeddingRetriever":
        self._chunks = chunks
        self._embeddings = self._model.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )
        return self

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        import numpy as np

        q = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q.T).flatten()
        order = np.argsort(scores)[::-1][:k]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in order]


def embeddings_available() -> bool:
    """True if sentence-transformers is importable and the model loads."""
    try:
        EmbeddingRetriever()
        return True
    except Exception:
        return False


def build_retriever(prefer: str = "auto") -> BaseRetriever:
    """Return a retriever. ``prefer`` is 'auto', 'embeddings' or 'tfidf'.

    In auto/embeddings mode any failure (package missing, model download
    blocked, no network) falls back to TF-IDF instead of crashing.
    """
    if prefer == "tfidf":
        return TfidfRetriever()
    if prefer in ("auto", "embeddings"):
        try:
            return EmbeddingRetriever()
        except Exception:
            if prefer == "embeddings":
                raise
            return TfidfRetriever()
    raise ValueError(f"Unknown retriever preference: {prefer}")
