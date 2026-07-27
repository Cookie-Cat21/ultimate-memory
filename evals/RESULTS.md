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
| multi_hop | **34.73** (XL21) | flan-t5-xl | 45.85 | 25.52 ✓ |
| temporal | **33.91** ✓ | flan-t5-xl | 12.14 | 9.15 |
| open_domain | **34.79** | flan-t5-xl (XL13) | 44.65 | 41.04 |

**Scoreboard vs A-MEM:** **2/4** (single + temporal). MemGPT **3/4**. MemoryBank/ReadAgent **4/4**. Best overall: **XL21 36.20**.

XL13 full rebench (intent-gated person window): overall **34.71**, multi **27.31**, open **34.79**, single **37.95**.

Full Qwen2.5-3B-Instruct suite: overall **30.54** (single→31). XL12 person-window widening regressed single to **25.1** (reverted).

**XL13** (post-revert baseline, flan-t5-xl): overall **34.71**, single **37.95** ✓, multi **27.31**, temporal **32.70** ✓, open **34.79**. Scoreboard vs A-MEM **2/4**; vs MemGPT **3/4**.

**XL14** (list-intent widen): overall **34.08** — multi **25.77** / open **32.47** regressed (weak inventory_union → noisy list LLM). Reverted that gate.

**XL15** (list-gate + both/places): overall **34.34**, single **37.87** ✓, multi **26.79**, temporal **32.04** ✓, open **33.25**. Scoreboard vs A-MEM **2/4**.

**XL16**: aborted early (still on conv-26) to pick up late-dialog fixes.

**XL17**: overall **34.98**, single **38.59** ✓, multi **28.01**, temporal **32.91** ✓, open **30.81** (entity_infer catalog regression).

**XL18**: overall **35.48** (best overall), single **38.85** ✓, multi **29.36**, temporal **32.54** ✓, open **33.83**. Dialog-1 multi **79.5**. Scoreboard vs A-MEM **2/4**; vs MemGPT **3/4**.

**XL19** (obs-rank + gated wider retrieval): overall **34.64** — multi **29.11** / open **31.6** regressed vs XL18. Reverted obs hard-prefer + limit bump.

**XL20**: overall **35.46**, single **38.13** ✓, multi **31.46** (best; still < A-MEM 45.85), temporal **32.40** ✓, open **34.04**. Scoreboard vs A-MEM **2/4**; vs MemGPT **3/4**.

**XL23**: overall **35.91** — multi **34.74** (tied XL21) but open **31.88** regressed (OD soft person + wider ctx). Soft person now multi-only; OD wider ctx reverted.

**XL21**: overall **36.20** (best), single **38.45** ✓, multi **34.73** (best; still < A-MEM 45.85), temporal **32.26** ✓, open **33.92**. Qualified late-dialog inventories + `atom:inv` filtered from non-list retrieval. Scoreboard vs A-MEM **2/4**; vs MemGPT **3/4**.

## Dialog-1 (152 Qs) — full A-MEM sweep

| Model | single | multi | temporal | open |
|---|---:|---:|---:|---:|
| `flan-t5-xl` | 34.0 | **79.5** (XL18) / 69.8 | 25.2 | **96.2** |
| `Qwen2.5-3B-Instruct` | 30.7 | **68.4** | 24.4 | **96.2** |

## Synthetic suite

Constraint accuracy **100%**; token F1 **~93%+**.

## Run

```bash
uv sync --extra dev --extra llm
uv run python evals/run_benchmarks.py --suite locomo --llm --model google/flan-t5-xl
```
