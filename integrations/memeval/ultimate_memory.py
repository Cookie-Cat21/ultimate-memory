"""ProsusAI MemEval adapter for Ultimate Memory.

Install ultimate-memory into the MemEval environment, then copy or symlink this
file into MemEval src/agents_memory/systems as ultimate_memory.py.
"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results

from ultimate_memory.config import (
    BasicMemoryConfig,
    DashboardConfig,
    Neo4jConfig,
    PathsConfig,
    QdrantConfig,
    RetrievalConfig,
    Settings,
)
from ultimate_memory.router import MemoryRouter

SYSTEM_INFO = {
    "architecture": (
        "local-first atomic bi-temporal memory with speaker-grounded propositions, "
        "entity/temporal planning, hybrid retrieval, and adaptive multi-hop evidence chains"
    ),
    "infrastructure": "SQLite FTS + optional Qdrant/Neo4j; adapter uses local fallback",
}


def _settings(root: Path) -> Settings:
    vault = root / "vault"
    private = root / "private"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=PathsConfig(repo_root=root, basic_memory_vault=vault, private_store=private),
        retrieval=RetrievalConfig(
            collection_name=f"memeval_{root.name}",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunk_chars=900,
            chunk_overlap=120,
            default_limit=12,
            bootstrap_token_budget_chars=6000,
        ),
        qdrant=QdrantConfig(url="http://127.0.0.1:1"),
        neo4j=Neo4jConfig(uri="bolt://127.0.0.1:1", user="neo4j", password="password"),
        basic_memory=BasicMemoryConfig(project="main", cli="basic-memory-missing"),
        dashboard=DashboardConfig(host="127.0.0.1", port=8787),
    )


def _ingest_dialogues(router: MemoryRouter, dialogues: list[dict], project_path: str) -> None:
    sessions: dict[str, list[dict]] = defaultdict(list)
    for turn in dialogues:
        sessions[str(turn.get("timestamp") or "unknown")].append(turn)

    for index, (timestamp, turns) in enumerate(sessions.items(), start=1):
        lines = ["---"]
        if timestamp and timestamp != "unknown":
            lines.extend([f"created_at: {timestamp}", f"session_date: {timestamp}"])
        lines.extend(["---", ""])
        for turn in turns:
            dia_id = str(turn.get("dia_id") or f"D{index}:0")
            speaker = str(turn.get("speaker") or "Unknown")
            turn_text = str(turn.get("text") or "").strip()
            lines.append(f"[{dia_id}] {speaker}: {turn_text}")
        router.ingest_log(
            client="memeval",
            session_id=f"session-{index}",
            transcript_or_path="\n".join(lines),
            project_path=project_path,
            tags=["memeval"],
        )


def _answer_with_shared_model(
    client: OpenAI,
    model: str,
    question: str,
    contexts: list[str],
) -> str:
    evidence = "\n\n".join(
        f"[Memory {index}] {memory_text}"
        for index, memory_text in enumerate(contexts[:16], start=1)
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=160,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied memory evidence. "
                    "Resolve multi-step relationships and dates when needed. "
                    "Give only the shortest complete answer and do not explain reasoning. "
                    "If the evidence is insufficient, say I don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nMemory evidence:\n{evidence}",
            },
        ],
    )
    return str(response.choices[0].message.content or "").strip()


def run(
    conv: dict,
    llm_model: str,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    client = OpenAI()
    dialogues = extract_dialogues(conv)

    with tempfile.TemporaryDirectory(prefix="ultimate-memory-memeval-") as temp:
        root = Path(temp)
        project_path = str(root / "project")
        router = MemoryRouter(_settings(root))
        _ingest_dialogues(router, dialogues, project_path)

        def answer_fn(question: str) -> str:
            result = router.answer(
                question,
                project_path=project_path,
                limit=24,
                use_llm=False,
                use_reader=False,
            )
            contexts = result.get("contexts_used") or []
            return _answer_with_shared_model(client, llm_model, question, contexts)

        return _qa_results(
            conv,
            answer_fn,
            run_judge,
            category_names=category_names,
            judge_fn=judge_fn,
        )
