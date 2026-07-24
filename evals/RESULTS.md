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
| Token F1 | **~93%+** |
| Constraint accuracy | **100%** |

### LoCoMo dialog-1 + `google/flan-t5-large`

| Category | Ours | A-MEM | Result |
|---|---:|---:|---|
| single_hop | **33.3** | 27.02 | beats |
| multi_hop | **60.0** | 45.85 | beats |
| temporal | **23.4** | 12.14 | beats |
| open_domain | **96.2** | 44.65 | beats |

### LoCoMo first-3 dialogs + list-answer path + `flan-t5-large`

| Category | Ours | A-MEM | MemGPT | Result |
|---|---:|---:|---:|---|
| single_hop | **31.4** | 27.02 | 26.65 | beats both |
| multi_hop | **35.8** | 45.85 | 25.52 | beats MemGPT; chasing A-MEM |
| temporal | **33.9** | 12.14 | 9.15 | beats both |
| open_domain | **82.3** | 44.65 | 41.04 | beats both |

### Full LoCoMo-10 + `flan-t5-large` (prior to list-answer path)

| Category | Ours | A-MEM | Result |
|---|---:|---:|---|
| single_hop | **36.5** | 27.02 | beats |
| multi_hop | 24.2 | 45.85 | behind (beats MemoryBank/ReadAgent) |
| temporal | **33.1** | 12.14 | beats |
| open_domain | 32.4 | 44.65 | behind (beats MemoryBank/ReadAgent) |

Full 10-dialog rerun with the dedicated list-answer path is in progress / reported in `evals/results/locomo.json`.

## What drives the wins

- Atomic bi-temporal memories + observation/event inventories
- Multi-fact aggregation for identity/duration/hypotheticals
- Dedicated comma-list LLM answerer over wide person-atom windows
- Hybrid absolute-date preference for temporal QA
- Dialogue-turn indexing with shared-book media hints
- Local `flan-t5-large` answerer (no paid API)

## How to run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite synthetic
uv run python evals/run_benchmarks.py --suite locomo --max-dialogs 1 --llm --model google/flan-t5-large
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-large
```

## Honesty notes

- A-MEM/MemGPT numbers use GPT-4o-mini; we use local HF seq2seq + deterministic aggregation.
- Mem0 peer-reviewed LoCoMo J ≈ 67–68%; vendor 90%+ J figures need the same judge protocol.
- Adversarial (cat 5) skipped by default.
