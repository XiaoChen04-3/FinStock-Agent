from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]


class AbstractVectorStore:
    def upsert(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def search(self, collection: str, query_text: str, top_k: int, threshold: float) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, collection: str, doc_id: str) -> None:
        raise NotImplementedError

    def delete_collection(self, collection: str) -> None:
        raise NotImplementedError


class ChromaVectorStore(AbstractVectorStore):
    def __init__(self) -> None:
        self._client = None
        self._embedding_function = None
        self._lock = Lock()
        self._data_path = settings.data_dir / "chroma"

    def upsert(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        if not text.strip():
            return
        chroma_collection = self._get_collection(collection)
        payload = dict(metadata or {})
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        chroma_collection.upsert(ids=[doc_id], documents=[text], metadatas=[payload])

    def search(self, collection: str, query_text: str, top_k: int, threshold: float) -> list[SearchResult]:
        if not query_text.strip():
            return []
        chroma_collection = self._get_collection(collection)
        if hasattr(chroma_collection, "count") and chroma_collection.count() == 0:
            return []
        raw = chroma_collection.query(
            query_texts=[query_text],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        results: list[SearchResult] = []
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = self._distance_to_similarity(distance)
            if similarity < threshold:
                continue
            results.append(
                SearchResult(
                    doc_id=str(doc_id),
                    text=str(document or ""),
                    similarity=similarity,
                    metadata=dict(metadata or {}),
                )
            )
        return results

    def delete(self, collection: str, doc_id: str) -> None:
        chroma_collection = self._get_collection(collection)
        chroma_collection.delete(ids=[doc_id])

    def delete_collection(self, collection: str) -> None:
        client = self._get_client()
        try:
            client.delete_collection(collection)
        except Exception:
            logger.debug("Ignoring missing Chroma collection during delete: %s", collection)

    def _get_collection(self, name: str):
        client = self._get_client()
        return client.get_or_create_collection(name=name, embedding_function=self._get_embedding_function())

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import chromadb
            except Exception as exc:  # pragma: no cover - depends on optional package
                raise RuntimeError("chromadb is not installed") from exc
            self._data_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._data_path))
            return self._client

    def _get_embedding_function(self):
        if self._embedding_function is not None:
            return self._embedding_function
        with self._lock:
            if self._embedding_function is not None:
                return self._embedding_function
            try:
                from chromadb.utils import embedding_functions
            except Exception as exc:  # pragma: no cover - depends on optional package
                raise RuntimeError("chromadb embedding functions are unavailable") from exc
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for Chroma embeddings")
            self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=get_config().models.embedding,
                api_base=settings.openai_base_url,
            )
            return self._embedding_function

    @staticmethod
    def _distance_to_similarity(distance: float | int | None) -> float:
        """将 Chroma 余弦距离转为相似度分数。

        Chroma 默认使用余弦距离，范围 [0, 2]（0 完全相同，2 完全相反）。
        转换公式：similarity = 1 - distance / 2，确保结果在 [0, 1]。
        """
        if distance is None:
            return 0.0
        value = float(distance)
        return max(0.0, min(1.0, 1.0 - value / 2.0))


_VECTOR_STORE: AbstractVectorStore | None = None


def get_vector_store() -> AbstractVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE
    backend = get_config().memory.vector_store.backend
    if backend != "chromadb":
        raise RuntimeError(f"Unsupported vector store backend: {backend}")
    _VECTOR_STORE = ChromaVectorStore()
    return _VECTOR_STORE


def reset_vector_store_for_tests() -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = None
