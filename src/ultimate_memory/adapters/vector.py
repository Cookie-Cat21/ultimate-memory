from __future__ import annotations

from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import QdrantConfig, RetrievalConfig
from ..models import AtomicMemory, MemoryChunk, SearchResult


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

    def upsert_chunks(self, chunks: Iterable[MemoryChunk], batch_size: int = 16) -> int:
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

    def upsert_atoms(self, atoms: Iterable[AtomicMemory], batch_size: int = 16) -> int:
        atom_list = list(atoms)
        if not atom_list or not self.ensure_collection():
            return 0
        total = 0
        for i in range(0, len(atom_list), batch_size):
            batch = atom_list[i : i + batch_size]
            vectors = self.embed([atom.text for atom in batch])
            points = [
                qmodels.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, atom.id)),
                    vector=vector,
                    payload=self._atom_payload(atom),
                )
                for atom, vector in zip(batch, vectors, strict=True)
            ]
            self.client().upsert(
                collection_name=self.retrieval.collection_name,
                points=points,
                wait=True,
            )
            total += len(points)
        return total

    @staticmethod
    def _atom_payload(atom: AtomicMemory) -> dict:
        tags = atom.metadata.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        return {
            "id": atom.id,
            "text": atom.text,
            "title": f"{atom.memory_type.value}: {atom.text[:72]}",
            "source_path": f"atom://{atom.id}",
            "memory_type": atom.memory_type.value,
            "kind": "atom",
            "project_path": atom.project_path,
            "tags": tags,
            "salience": atom.salience,
            "valid_from": atom.valid_from,
            "valid_until": atom.valid_until,
            "superseded_by": atom.superseded_by,
            "access_count": atom.access_count,
            "entities": atom.entities,
            "created_at": atom.created_at,
        }

    @staticmethod
    def _search_result_from_payload(payload: dict, *, score: float) -> SearchResult:
        if payload.get("kind") == "atom":
            return SearchResult(
                id=str(payload.get("id", "")),
                text=str(payload.get("text", "")),
                title=str(payload.get("title", "Untitled")),
                source_path=payload.get("source_path") or f"atom://{payload.get('id', '')}",
                memory_type=str(payload.get("memory_type", "note")),
                score=score,
                provenance={
                    "source": "atomic-memory",
                    "vector": True,
                    "salience": payload.get("salience"),
                    "valid_from": payload.get("valid_from"),
                    "valid_until": payload.get("valid_until"),
                    "superseded_by": payload.get("superseded_by"),
                    "access_count": payload.get("access_count"),
                    "entities": payload.get("entities"),
                    "project_path": payload.get("project_path"),
                    "payload": payload,
                },
            )
        return SearchResult(
            id=str(payload.get("id", "")),
            text=str(payload.get("text", "")),
            title=str(payload.get("title", "Untitled")),
            source_path=payload.get("source_path"),
            memory_type=str(payload.get("memory_type", "note")),
            score=score,
            provenance={"source": "qdrant", "payload": payload},
        )

    def search(
        self,
        query: str,
        limit: int = 8,
        tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[SearchResult]:
        if not query.strip() or not self.ensure_collection():
            return []
        vector = self.embed([query])[0]

        conditions: list[qmodels.FieldCondition] = []
        if tags:
            conditions.extend(
                qmodels.FieldCondition(key="tags", match=qmodels.MatchValue(value=tag))
                for tag in tags
            )
        if memory_types:
            conditions.append(
                qmodels.FieldCondition(
                    key="memory_type",
                    match=qmodels.MatchAny(any=memory_types),
                )
            )
        query_filter = qmodels.Filter(must=conditions) if conditions else None

        response = self.client().query_points(
            collection_name=self.retrieval.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            if (
                not include_superseded
                and payload.get("kind") == "atom"
                and payload.get("valid_until")
            ):
                continue
            results.append(
                self._search_result_from_payload(payload, score=float(point.score))
            )
        return results
