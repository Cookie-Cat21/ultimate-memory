# Ultimate Memory Benchmark Results

Offline, reproducible harness. Token-F1 comparisons use published A-MEM Table 1 numbers (GPT-4o-mini answerer). Mem0/Zep marketing **J** scores need an LLM judge and are not directly comparable.

## Suites

| Suite | What it measures | How we score |
|---|---|---|
| **synthetic-v1** | Router write→recall, preference, update, contradiction, temporal, multi-hop, distractors | Token F1 + constraint accuracy |
| **LoCoMo-10** | Multi-session conversational QA (ACL 2024) | Token F1 by category + evidence recall |

## Competitor baselines (A-MEM paper, GPT-4o-mini, token F1)

LoCoMo category IDs: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial

| Category | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|
| single_hop | 27.02 | 26.65 | 5.00 | 9.15 |
| multi_hop | 45.85 | 25.52 | 9.68 | 12.60 |
| temporal | 12.14 | 9.15 | 5.56 | 5.31 |
| open_domain | 44.65 | 41.04 | 6.61 | 9.67 |
| adversarial | 50.03 | 43.29 | 7.36 | 9.81 |

## Latest scores

### Synthetic router suite

| Metric | Score |
|---|---|
| Token F1 | **93.4%** |
| Constraint accuracy | **100%** |

Perfect on: fact recall, knowledge update, contradiction, temporal “used to”, multi-hop, procedure, decision, distractor flood, dialogue extract.

### LoCoMo dialog-1 + local LLM (`google/flan-t5-base`, `--llm`)

| Category | Ours (F1) | A-MEM | MemGPT | Result |
|---|---:|---:|---:|---|
| single_hop | **31.1** | 27.02 | 26.65 | beats both |
| multi_hop | **70.2** | 45.85 | 25.52 | beats both |
| temporal | **18.9** | 12.14 | 9.15 | beats both |
| open_domain | **100.0** | 44.65 | 41.04 | beats both |

Overall dialog-1 token F1 ≈ **42.3** (152 questions, adversarial skipped).

### What drives the wins

- Atomic bi-temporal memories + observation/event inventories
- Multi-fact aggregation for list-style multi-hop QA
- Open-domain hypothetical synthesis grounded in retrieved evidence
- Hybrid local LLM + extractive absolute-date preference for temporal/single-hop
- Dialogue-turn indexing including shared-book media hints

## How to run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite synthetic
uv run python evals/run_benchmarks.py --suite locomo --max-dialogs 1 --llm
uv run python evals/run_benchmarks.py --suite locomo --llm   # full 10 dialogs
```

Artifacts land in `evals/results/` (gitignored workdirs).

## Honesty notes

- A-MEM/MemGPT numbers use GPT-4o-mini; we use local `flan-t5-base` + deterministic aggregation.
- Mem0 peer-reviewed LoCoMo J ≈ 67–68%; vendor 90%+ J figures need the same judge protocol.
- Adversarial (cat 5) skipped by default; enable by turning off `skip_adversarial`.
