#!/usr/bin/env python3
"""Leakage-resistant LoCoMo evaluation.

Rules:
- ingest raw dialogue sessions only
- never ingest observation/event_summary/session_summary annotations
- never pass gold QA category to the memory router
- no media-title lookup table or question-specific hints
- report retrieval evidence metrics separately from answer token-F1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ultimate_memory.answer import tokenize_f1
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results-clean"

CAT_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def make_settings(work: Path) -> Settings:
    vault = work / "vault"
    private = work / "private"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=PathsConfig(repo_root=work, basic_memory_vault=vault, private_store=private),
        retrieval=RetrievalConfig(
            collection_name=f"clean_{work.name}",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunk_chars=900,
            chunk_overlap=120,
            default_limit=8,
            bootstrap_token_budget_chars=6000,
        ),
        qdrant=QdrantConfig(url="http://127.0.0.1:1"),
        neo4j=Neo4jConfig(uri="bolt://127.0.0.1:1", user="neo4j", password="password"),
        basic_memory=BasicMemoryConfig(project="main", cli="basic-memory-missing"),
        dashboard=DashboardConfig(host="127.0.0.1", port=8787),
    )


def raw_session_transcript(conversation: dict[str, Any], session_key: str) -> str:
    turns = conversation.get(session_key) or []
    when = conversation.get(f"{session_key}_date_time")
    lines = ["---"]
    if when:
        lines.extend([f"created_at: {when}", f"session_date: {when}"])
    lines.extend(["---", ""])
    for turn in turns:
        speaker = turn.get("speaker", "Speaker")
        dia = turn.get("dia_id", "")
        text = str(turn.get("text") or "").strip()
        prefix = f"[{dia}] " if dia else ""
        lines.append(f"{prefix}{speaker}: {text}")
    return "\n".join(lines)


def evidence_recall(contexts: list[str], evidence_ids: list[str], conversation: dict[str, Any]) -> float:
    if not evidence_ids:
        return 1.0
    blob = "\n".join(contexts).lower()
    hits = 0
    for evidence_id in evidence_ids:
        target = None
        for key, turns in conversation.items():
            if not re.fullmatch(r"session_\d+", key) or not isinstance(turns, list):
                continue
            for turn in turns:
                if turn.get("dia_id") == evidence_id:
                    target = str(turn.get("text") or "").strip()
                    break
            if target:
                break
        if evidence_id.lower() in blob:
            hits += 1
        elif target and target.lower()[:48] in blob:
            hits += 1
    return hits / len(evidence_ids)



def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def gold_token_coverage(contexts: list[str], golds: list[str]) -> float:
    """Best fraction of gold answer tokens present anywhere in retrieved contexts."""
    context_tokens = _token_set("\n".join(contexts))
    best = 0.0
    for gold in golds:
        gold_tokens = _token_set(gold)
        if not gold_tokens:
            continue
        best = max(best, len(context_tokens & gold_tokens) / len(gold_tokens))
    return best


def expected_plan_kind(category: str) -> str:
    if category == "multi_hop":
        return "multi_hop"
    if category == "temporal":
        return "temporal"
    return "single_hop"


def run(
    *,
    start_dialog: int = 0,
    max_dialogs: int | None = None,
    max_questions: int | None = None,
    use_llm: bool = False,
) -> dict:
    data = json.loads((DATA / "locomo10.json").read_text(encoding="utf-8"))
    data = data[max(start_dialog, 0):]
    if max_dialogs is not None:
        data = data[:max_dialogs]

    work_root = RESULTS / "work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    scores: dict[str, list[float]] = defaultdict(list)
    evidence_scores: list[float] = []
    evidence_by_category: dict[str, list[float]] = defaultdict(list)
    token_coverage_scores: list[float] = []
    token_coverage_by_category: dict[str, list[float]] = defaultdict(list)
    planner_total = 0
    planner_correct = 0
    planner_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    asked = 0
    started = time.perf_counter()

    for sample in data:
        sample_id = str(sample.get("sample_id") or "conv")
        work = work_root / re.sub(r"[^\w.-]+", "_", sample_id)
        router = MemoryRouter(make_settings(work))
        conversation = sample["conversation"]

        for key in sorted(conversation):
            if not re.fullmatch(r"session_\d+", key):
                continue
            router.ingest_log(
                client="locomo-clean",
                session_id=f"{sample_id}-{key}",
                transcript_or_path=raw_session_transcript(conversation, key),
                project_path=str(work / "project"),
                tags=["locomo-clean", sample_id],
            )

        for qa in sample.get("qa", []):
            if max_questions is not None and asked >= max_questions:
                break
            category = CAT_NAMES.get(int(qa.get("category", 0)), "unknown")
            if category == "adversarial":
                continue
            gold = qa["answer"]
            golds = gold if isinstance(gold, list) else [str(gold)]

            result = router.answer(
                qa["question"],
                project_path=str(work / "project"),
                limit=20,
                use_llm=use_llm,
            )
            score = tokenize_f1(result["answer"], golds)
            scores[category].append(score)
            contexts_used = result.get("contexts_used") or []
            evidence_score = evidence_recall(
                contexts_used, qa.get("evidence") or [], conversation
            )
            evidence_scores.append(evidence_score)
            evidence_by_category[category].append(evidence_score)

            coverage = gold_token_coverage(contexts_used, [str(g) for g in golds])
            token_coverage_scores.append(coverage)
            token_coverage_by_category[category].append(coverage)

            planned_kind = str((result.get("query_plan") or {}).get("kind") or "unknown")
            expected_kind = expected_plan_kind(category)
            planner_total += 1
            planner_correct += int(planned_kind == expected_kind)
            planner_confusion[expected_kind][planned_kind] += 1
            asked += 1

        if max_questions is not None and asked >= max_questions:
            break

    by_category = {
        category: {
            "n": len(values),
            "token_f1": round(100 * sum(values) / len(values), 2) if values else 0.0,
        }
        for category, values in sorted(scores.items())
    }
    all_scores = [score for values in scores.values() for score in values]
    retrieval_by_category = {
        category: {
            "evidence_recall": round(100 * sum(values) / len(values), 2) if values else 0.0,
            "gold_token_coverage": round(
                100 * sum(token_coverage_by_category[category])
                / len(token_coverage_by_category[category]),
                2,
            )
            if token_coverage_by_category[category]
            else 0.0,
        }
        for category, values in sorted(evidence_by_category.items())
    }
    return {
        "benchmark": "locomo10-clean",
        "protocol": "raw-dialogue-only/no-gold-category/no-annotation-summaries",
        "questions": asked,
        "start_dialog": start_dialog,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "overall_token_f1": round(100 * sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
        "evidence_recall": round(100 * sum(evidence_scores) / len(evidence_scores), 2)
        if evidence_scores else 0.0,
        "gold_token_coverage": round(
            100 * sum(token_coverage_scores) / len(token_coverage_scores), 2
        )
        if token_coverage_scores else 0.0,
        "planner_kind_accuracy": round(100 * planner_correct / planner_total, 2)
        if planner_total else 0.0,
        "planner_confusion": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(planner_confusion.items())
        },
        "retrieval_by_category": retrieval_by_category,
        "by_category": by_category,
        "use_llm": use_llm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-dialog", type=int, default=0)
    parser.add_argument("--max-dialogs", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.max_dialogs = args.max_dialogs or 1
        args.max_questions = args.max_questions or 40
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run(
        start_dialog=args.start_dialog,
        max_dialogs=args.max_dialogs,
        max_questions=args.max_questions,
        use_llm=args.llm,
    )
    (RESULTS / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
