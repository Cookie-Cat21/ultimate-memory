"""Compatibility re-export for evaluation scripts."""

from ultimate_memory.eval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = ["ndcg_at_k", "recall_at_k", "reciprocal_rank"]
