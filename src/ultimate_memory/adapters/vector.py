from __future__ import annotations

from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import QdrantConfig, RetrievalConfig
from ..models import MemoryChunk, SearchResult


class VectorAdapter:
    def __init__(self, config: QdrantConfig, retrieval: RetrievalConfig) -> None:
        self.config = config
        self.retrieval = retrieval
        self._client: QdrantClient | None = None
        self._embedding_model = None
        self._embedding_size = 384

    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.url,
                timeout=3,
                check_compatibility=False,
            )
        return self._client

    def is_available(self) -> bool:
        try:
            self.client().get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self) -> bool:
        if not self.is_available():
            return False
        collections = self.client().get_collections().collections
        if any(collection.name == self.retrieval.collection_name for collection in collections):
            return True
        self.client().create_collection(
            collection_name=self.retrieval.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._embedding_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        # FastEmbed model is imported lazily to keep fallback mode light.
        if self._embedding_model is None:
            from fastembed import TextEmbedding
            self._embedding_model = TextEmbedding(model_name=self.retrieval.embedding_model)
        return [list(vector) for vector in self._embedding_model.embed(texts)]

    def upsert_chunks(self, chunks: Iterable[MemoryChunk], batch_size: int = 32) -> int:
        chunk_list = list(chunks)
        if not chunk_list or not self.ensure_collection():
            return 0
        total = 0
        for i in range(0, len(chunk_list), batch_size):
            batch = chunk_list[i : i + batch_size]
            vectors = self.embed([chunk.text for chunk in batch])
            points = [
                qmodels.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.id)),
                    vector=vector,
                    payload={
                        "id": chunk.id,
                        "text": chunk.text,
                        "title": chunk.title,
                        "source_path": chunk.source_path,
                        "memory_type": chunk.memory_type.value,
                        "project_path": chunk.project_path,
                        "tags": chunk.tags,
                        "created_at": chunk.created_at,
                        **chunk.metadata,
                    },
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client().upsert(
                collection_name=self.retrieval.collection_name,
                points=points,
                wait=True,
            )
            total += len(points)
        return total

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if not query.strip() or not self.ensure_collection():
            return []
        vector = self.embed([query])[0]
        response = self.client().query_points(
            collection_name=self.retrieval.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    id=str(payload.get("id", point.id)),
                    text=str(payload.get("text", "")),
                    title=str(payload.get("title", "Untitled")),
                    source_path=payload.get("source_path"),
                    memory_type=str(payload.get("memory_type", "note")),
                    score=float(point.score),
                    provenance={"source": "qdrant", "payload": payload},
                )
            )
        return results
