#!/usr/bin/env python3
"""Fail CI if the clean evaluation path regains benchmark leakage."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_absent(path: Path, needles: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [needle for needle in needles if needle in text]


def main() -> None:
    failures: list[str] = []

    clean_eval = ROOT / "evals" / "run_clean_benchmarks.py"
    forbidden_eval = [
        'sample.get("observation")',
        'sample.get("event_summary")',
        'sample.get("session_summary")',
        "category=category",
        "category=cat_name",
        "_MEDIA_TITLE_HINTS",
    ]
    for needle in require_absent(clean_eval, forbidden_eval):
        failures.append(f"{clean_eval}: forbidden leakage marker {needle!r}")

    router = ROOT / "src" / "ultimate_memory" / "router.py"
    forbidden_router = [
        'category == "multi_hop"',
        'category == "single_hop"',
        '(category or "").strip().lower()',
    ]
    for needle in require_absent(router, forbidden_router):
        failures.append(f"{router}: gold-category routing marker {needle!r}")

    planner = ROOT / "src" / "ultimate_memory" / "planner.py"
    forbidden_planner = ["LoCoMo", "Joanna", "Jon & Gina", "UNO", "Mafia", "Charlotte's Web"]
    for needle in require_absent(planner, forbidden_planner):
        failures.append(f"{planner}: benchmark-specific marker {needle!r}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("benchmark integrity checks passed")


if __name__ == "__main__":
    main()
