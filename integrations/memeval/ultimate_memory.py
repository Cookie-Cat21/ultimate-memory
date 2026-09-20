"""ProsusAI/MemEval adapter for Ultimate Memory.

Copy this file into MemEval's ``src/agents_memory/systems/ultimate_memory.py``
and install Ultimate Memory in the same environment. MemEval will discover it
automatically through its system registry.

The adapter intentionally uses the benchmark-provided answer model and
text-embedding-3-small so comparisons with PropMem use the same reader and
embedding family rather than Ultimate Memory's local extractive reader.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from openai import OpenAI

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
from ultimate_memory.dates import parse_loose_date
from ultimate_memory.local_semantic import LocalSemanticIndex
from ultimate_memory.models import CompiledMemory
from ultimate_memory.router import MemoryRouter


SYSTEM_INFO = {
    "architecture": (
        "atomic bi-temporal memory + entity-scoped hybrid retrieval + "
        "conversation windows + adaptive multi-hop planning"
    ),
    "infrastructure": "SQLite FTS + in-process OpenAI semantic retrieval; no external DB required",
}


COMPILE_PROMPT = """Compile durable atomic memories from this raw conversation session.

Conversation participants: {participants}
Session date: {session_date}

Raw turns:
{turns}

Rules:
1. Extract only information supported by the turns. Never invent or infer unstated facts.
2. Each memory must be atomic: one durable fact, preference, decision, procedure, event,
   relationship, status, ownership fact, activity, or plan.
3. Name the subject explicitly. Replace first-person pronouns with the speaker name.
4. Preserve exact names, places, titles, dates, quantities, and temporal qualifiers.
5. Keep separate facts separate; do not merge unrelated information.
6. Include the dialogue IDs that directly support each memory.
7. entities must contain the people/organizations/places directly involved.
8. memory_type must be one of: fact, preference, decision, procedure.
9. Do not extract greetings, compliments, filler, or questions that contain no answer.
10. Confidence is 0-1 and should reflect how explicitly the memory is stated.

Return JSON:
{{"memories": [
  {{
    "text": "explicit atomic memory",
    "memory_type": "fact",
    "entities": ["Entity"],
    "source_dia_ids": ["D1:2"],
    "confidence": 0.95
  }}
]}}
"""


ANSWER_PROMPT = """Answer the question using ONLY the memory evidence below.

Known conversation participants: {participants}

Evidence:
{evidence}

Question: {question}

Rules:
1. Reason from the evidence, including across multiple evidence items when needed.
2. Keep facts attached to the correct person; do not transfer another person's facts.
3. For temporal questions, use the dates/timestamps in the evidence and resolve relative dates.
4. For lists or multi-part questions, include every supported item and deduplicate them.
5. For questions asking what would/could/likely happen, make the minimal inference supported by evidence.
6. Give a direct compact answer, not an explanation. Prefer exact words from the evidence.
7. If the evidence does not support an answer, return "None".
8. Answer in the same language as the question.

Return JSON exactly as:
{{"answer": "direct compact answer"}}
"""


def _settings(root: Path) -> Settings:
    vault = root / "vault"
    private = root / "private"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=PathsConfig(
            repo_root=root,
            basic_memory_vault=vault,
            private_store=private,
        ),
        retrieval=RetrievalConfig(
            collection_name="memeval",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunk_chars=900,
            chunk_overlap=120,
            default_limit=12,
            bootstrap_token_budget_chars=6000,
        ),
        qdrant=QdrantConfig(url="http://127.0.0.1:1"),
        neo4j=Neo4jConfig(
            uri="bolt://127.0.0.1:1",
            user="neo4j",
            password="disabled",
        ),
        basic_memory=BasicMemoryConfig(project="memeval", cli="basic-memory-missing"),
        dashboard=DashboardConfig(host="127.0.0.1", port=8787),
    )


def _session_text(conversation: dict, session_key: str) -> str:
    turns = conversation.get(session_key) or []
    when = conversation.get(f"{session_key}_date_time") or ""
    lines = ["---"]
    if when:
        lines.extend([f"created_at: {when}", f"session_date: {when}"])
    lines.extend(["---", ""])
    for turn in turns:
        speaker = str(turn.get("speaker") or "Speaker")
        dia_id = str(turn.get("dia_id") or "")
        text = str(turn.get("text") or "").strip()
        prefix = f"[{dia_id}] " if dia_id else ""
        lines.append(f"{prefix}{speaker}: {text}")
    return "\n".join(lines)


def _compile_session(
    client: OpenAI,
    model: str,
    *,
    participants: list[str],
    session_date: str,
    session_text: str,
) -> list[CompiledMemory]:
    turn_lines = [
        line
        for line in session_text.splitlines()
        if line.strip().startswith("[D")
    ]
    if not turn_lines:
        return []
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": COMPILE_PROMPT.format(
                    participants=", ".join(participants) or "unknown",
                    session_date=session_date or "unknown",
                    turns="\n".join(turn_lines),
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=4096,
    )
    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    raw_memories = payload.get("memories") or payload.get("facts") or []
    if not isinstance(raw_memories, list):
        return []

    compiled: list[CompiledMemory] = []
    for raw in raw_memories[:60]:
        if not isinstance(raw, dict):
            continue
        try:
            compiled.append(CompiledMemory.model_validate(raw))
        except Exception:
            continue
    return compiled


def _participants(conversation: dict) -> list[str]:
    names: list[str] = []
    for key in ("speaker_a", "speaker_b"):
        value = str(conversation.get(key) or "").strip()
        if value and value not in names:
            names.append(value)
    if names:
        return names
    for key, turns in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(turns, list):
            continue
        for turn in turns:
            speaker = str(turn.get("speaker") or "").strip()
            if speaker and speaker not in names:
                names.append(speaker)
    return names[:8]


def run(
    conv: dict,
    llm_model: str,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conversation = conv["conversation"]
    participants = _participants(conversation)

    with TemporaryDirectory(prefix="ultimate-memory-memeval-") as temp_dir:
        root = Path(temp_dir)
        project_path = str(root / "conversation")
        router = MemoryRouter(_settings(root))

        # Use the exact embedding family from the MemEval comparison rig.
        def openai_embed(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = client.embeddings.create(
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=texts,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [list(item.embedding) for item in ordered]

        router._local_semantic = LocalSemanticIndex(openai_embed)

        previous_semantic = os.environ.get("ULTIMATE_MEMORY_LOCAL_SEMANTIC")
        os.environ["ULTIMATE_MEMORY_LOCAL_SEMANTIC"] = "1"
        try:
            session_keys = sorted(
                (
                    key
                    for key in conversation
                    if key.startswith("session_") and not key.endswith("_date_time")
                ),
                key=lambda key: int(key.split("_")[1]),
            )
            compiled_total = 0
            for session_key in session_keys:
                raw_session = _session_text(conversation, session_key)
                router.ingest_log(
                    client="memeval",
                    session_id=session_key,
                    transcript_or_path=raw_session,
                    project_path=project_path,
                    tags=["memeval"],
                )

                session_date = str(
                    conversation.get(f"{session_key}_date_time") or ""
                )
                compiled = _compile_session(
                    client,
                    llm_model,
                    participants=participants,
                    session_date=session_date,
                    session_text=raw_session,
                )
                parsed_date = parse_loose_date(session_date)
                event_time = parsed_date.isoformat() if parsed_date else None
                outcome = router.ingest_compiled_memories(
                    compiled,
                    session_id=session_key,
                    project_path=project_path,
                    event_time=event_time,
                    compiler=f"memeval:{llm_model}",
                )
                compiled_total += int(outcome.get("created") or 0)

            print(
                f"    Ingested: sessions={len(session_keys)}, "
                f"compiled_memories={compiled_total}"
            )

            def answer_fn(question: str) -> str:
                result = router.answer(
                    question,
                    project_path=project_path,
                    limit=16,
                    use_llm=False,
                )
                contexts: list[str] = []
                seen: set[str] = set()
                for raw in result.get("contexts_used") or []:
                    text = str(raw).strip()
                    key = text.casefold()
                    if not text or key in seen:
                        continue
                    seen.add(key)
                    contexts.append(text)
                    if len(contexts) >= 20:
                        break

                evidence = "\n\n".join(
                    f"[Memory {index}] {text}"
                    for index, text in enumerate(contexts, start=1)
                )
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": ANSWER_PROMPT.format(
                                participants=", ".join(participants) or "unknown",
                                evidence=evidence or "(no evidence retrieved)",
                                question=question,
                            ),
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=512,
                )
                content = response.choices[0].message.content or "{}"
                try:
                    payload = json.loads(content)
                    answer = str(payload.get("answer") or "").strip()
                except json.JSONDecodeError:
                    answer = content.strip()
                return answer or "None"

            return _qa_results(
                conv,
                answer_fn,
                run_judge,
                category_names=category_names,
                judge_fn=judge_fn,
            )
        finally:
            if previous_semantic is None:
                os.environ.pop("ULTIMATE_MEMORY_LOCAL_SEMANTIC", None)
            else:
                os.environ["ULTIMATE_MEMORY_LOCAL_SEMANTIC"] = previous_semantic
