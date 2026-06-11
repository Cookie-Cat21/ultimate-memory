from __future__ import annotations

import re
from typing import Iterable

from neo4j import GraphDatabase

from ..config import Neo4jConfig
from ..models import MemoryChunk

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class GraphAdapter:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver = None

    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                connection_timeout=3,
            )
        return self._driver

    def is_available(self) -> bool:
        try:
            with self.driver().session() as session:
                session.run("RETURN 1").single()
            return True
        except Exception:
            return False

    def ensure_schema(self) -> bool:
        if not self.is_available():
            return False
        with self.driver().session() as session:
            session.run("CREATE CONSTRAINT memory_entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT memory_source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE")
        return True

    def upsert_chunks(self, chunks: Iterable[MemoryChunk], batch_size: int = 64) -> int:
        if not self.ensure_schema():
            return 0
        chunk_list = list(chunks)
        total = 0
        with self.driver().session() as session:
            for i in range(0, len(chunk_list), batch_size):
                batch = chunk_list[i : i + batch_size]
                sources = [
                    {"source_id": c.source_path, "title": c.title,
                     "memory_type": c.memory_type.value, "source_path": c.source_path}
                    for c in batch
                ]
                session.run(
                    """
                    UNWIND $sources AS s
                    MERGE (src:Source {id: s.source_id})
                    SET src.title = s.title, src.memory_type = s.memory_type, src.path = s.source_path
                    """,
                    sources=sources,
                )
                links = [
                    {"source_id": c.source_path, "entity": entity}
                    for c in batch
                    for entity in sorted(set(WIKILINK_RE.findall(c.text)))
                ]
                if links:
                    session.run(
                        """
                        UNWIND $links AS link
                        MERGE (e:Entity {name: link.entity})
                        WITH e, link
                        MATCH (s:Source {id: link.source_id})
                        MERGE (s)-[:MENTIONS]->(e)
                        """,
                        links=links,
                    )
                total += len(batch)
        return total

    def add_reflection(
        self,
        *,
        title: str,
        memory_type: str,
        text: str,
        source_refs: list[str],
    ) -> bool:
        if not self.ensure_schema():
            return False
        with self.driver().session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $title})
                SET e.memory_type = $memory_type, e.text = $text
                """,
                title=title,
                memory_type=memory_type,
                text=text,
            )
            for ref in source_refs:
                session.run(
                    """
                    MERGE (s:Source {id: $ref})
                    WITH s
                    MATCH (e:Entity {name: $title})
                    MERGE (e)-[:SUPPORTED_BY]->(s)
                    """,
                    ref=ref,
                    title=title,
                )
        return True

    def supersede(self, old_ref: str, new_fact: str, reason: str, source_refs: list[str]) -> bool:
        if not self.ensure_schema():
            return False
        with self.driver().session() as session:
            session.run(
                """
                MERGE (old:Entity {name: $old_ref})
                MERGE (new:Entity {name: $new_fact})
                SET new.text = $new_fact
                MERGE (new)-[r:SUPERSEDES]->(old)
                SET r.reason = $reason, r.source_refs = $source_refs
                """,
                old_ref=old_ref,
                new_fact=new_fact,
                reason=reason,
                source_refs=source_refs,
            )
        return True

    _DEPTH_QUERIES = {
        1: "MATCH p=(e)-[*0..1]-(n) WHERE toLower(e.name) CONTAINS toLower($topic) OR toLower(coalesce(e.title, '')) CONTAINS toLower($topic) RETURN p LIMIT 25",
        2: "MATCH p=(e)-[*0..2]-(n) WHERE toLower(e.name) CONTAINS toLower($topic) OR toLower(coalesce(e.title, '')) CONTAINS toLower($topic) RETURN p LIMIT 25",
        3: "MATCH p=(e)-[*0..3]-(n) WHERE toLower(e.name) CONTAINS toLower($topic) OR toLower(coalesce(e.title, '')) CONTAINS toLower($topic) RETURN p LIMIT 25",
    }

    def query(self, entity_or_topic: str, depth: int = 1) -> list[dict]:
        if not entity_or_topic.strip() or not self.ensure_schema():
            return []
        depth = max(1, min(depth, 3))
        with self.driver().session() as session:
            records = session.run(
                self._DEPTH_QUERIES[depth],
                topic=entity_or_topic,
            )
            output: list[dict] = []
            for record in records:
                path = record["p"]
                output.append(
                    {
                        "nodes": [dict(node) | {"labels": list(node.labels)} for node in path.nodes],
                        "relationships": [
                            {"type": rel.type, "properties": dict(rel)} for rel in path.relationships
                        ],
                    }
                )
        return output

