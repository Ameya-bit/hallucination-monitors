# hallucination-monitors

EE 456 honors option (Penn State, Fall 2026). Do confident-but-wrong LLM answers leave an
internal activation signature that current hallucination monitors miss?

## Status

- **Gate A pilot** (`src/pilot/`): Qwen3-0.6B, closed-book TriviaQA, greedy decoding.
  Records first-answer-token probability, correctness, and residual-stream activations at
  every layer for the last prompt token and the first answer token. Pass criterion: at least
  150 wrong answers with first-token probability >= 0.8.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python src/pilot/run_pilot.py --config configs/pilot_qwen3_0.6b.json
```

Runs on Apple Silicon (MPS), CUDA, or CPU. Activations and raw data are git-ignored.

## Correctness labels

`contains_match` (any gold alias appears as a token span in the prediction) is the primary
label. `exact_match` is also recorded. Both are lexical; an LLM-judge audit on ~200 items is
planned, following Janiak et al. (EMNLP 2025).

## Layout

```
configs/      run configs (model, dataset, thresholds)
src/pilot/    Gate A pilot: generation + activation capture
results/      summaries and per-question records (committed)
data/         raw datasets and activations (ignored)
```
