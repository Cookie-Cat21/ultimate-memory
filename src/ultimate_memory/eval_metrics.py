"""Retrieval-only metrics for memory-system evaluation."""
from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for index, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    def dcg(items: Sequence[str]) -> float:
        total = 0.0
        for index, item in enumerate(items, start=1):
            if item in relevant_ids:
                total += 1.0 / math.log2(index + 1)
        return total

    actual = dcg(ranked_ids[:k])
    ideal_len = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    return actual / ideal if ideal else 1.0
