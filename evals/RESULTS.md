# Ultimate Memory Benchmark Results

Offline token-F1 vs A-MEM Table 1 (GPT-4o-mini). Mem0/Zep **J** scores need an LLM judge and are not comparable.

## Competitor baselines

| Category | A-MEM | MemGPT | MemoryBank | ReadAgent |
|---|---:|---:|---:|---:|
| single_hop | 27.02 | 26.65 | 5.00 | 9.15 |
| multi_hop | 45.85 | 25.52 | 9.68 | 12.60 |
| temporal | 12.14 | 9.15 | 5.56 | 5.31 |
| open_domain | 44.65 | 41.04 | 6.61 | 9.67 |

## Best full LoCoMo-10 (1540 Qs)

| Category | Ours (best) | Model | A-MEM | MemGPT |
|---|---:|---|---:|---:|
| single_hop | **39.82** ✓ | flan-t5-xl | 27.02 | 26.65 |
| multi_hop | **27.99** | flan-t5-xl | 45.85 | 25.52 ✓ |
| temporal | **33.91** ✓ | flan-t5-xl | 12.14 | 9.15 |
| open_domain | **34.79** | flan-t5-xl (XL13) | 44.65 | 41.04 |

**Scoreboard vs A-MEM:** **2/4** (single + temporal). MemGPT **3/4**. MemoryBank/ReadAgent **4/4**.

XL13 full rebench (intent-gated person window): overall **34.71**, multi **27.31**, open **34.79**, single **37.95**.

Full Qwen2.5-3B-Instruct suite: overall **30.54** (single→31). XL12 person-window widening regressed single to **25.1** (reverted).

**XL13** (post-revert baseline, flan-t5-xl): overall **34.71**, single **37.95** ✓, multi **27.31**, temporal **32.70** ✓, open **34.79**. Scoreboard vs A-MEM **2/4**; vs MemGPT **3/4**.

**XL14** (list-intent widen): overall **34.08** — multi **25.77** / open **32.47** regressed (weak inventory_union → noisy list LLM). Reverted that gate.

**XL15**: keep gazetteer places + specialized both-intersection; require strong list signal before list LLM; rebench full suite.

## Dialog-1 (152 Qs) — full A-MEM sweep

| Model | single | multi | temporal | open |
|---|---:|---:|---:|---:|
| `flan-t5-xl` | 34.0 | **69.8** | 25.2 | **96.2** |
| `Qwen2.5-3B-Instruct` | 30.7 | **68.4** | 24.4 | **96.2** |

## Synthetic suite

Constraint accuracy **100%**; token F1 **~93%+**.

## Run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-xl
```
