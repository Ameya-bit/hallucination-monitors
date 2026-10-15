# hallucination-monitors

EE 456 honors option (Penn State, Fall 2026). Do confident-but-wrong LLM answers leave an
internal activation signature that current hallucination monitors miss?

## Status

- **Gate A pilot** (`src/pilot/`): Qwen3-0.6B, closed-book TriviaQA, greedy decoding.
  Records first-answer-token probability, correctness, and residual-stream activations at
  every layer for the last prompt token and the first answer token. Pass criterion: at least
  150 wrong answers with first-token probability >= 0.8.

## Gate A results (Oct 15, 2026)

TriviaQA validation, first 2,000 questions, greedy decoding, 4-shot terse-answer prompt, thinking off.
"Confident" = first content-token probability >= 0.8. AUROC is 5-fold out-of-fold, L2 logistic
regression on the standardized residual stream at the last prompt token.

| Model | Acc. | Confident | Confident and wrong | Token-prob AUROC (conf. subset) | Probe AUROC (conf. subset) | Best layer |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 13.9% | 637 | 531 | 0.55 | 0.79 [0.74, 0.83] | 18 |
| Qwen3-4B | 39.8% | 1,057 | 405 | 0.68 | 0.76 [0.74, 0.79] | 24 |

Both runs pass Gate A (>= 150 confident errors) and H1 (probe CI above the token-probability
baseline on the confident subset). Known issues: labels are lexical and not yet judge-audited; the
correctness-vs-confidence axis cosine in `metrics.json` is confounded by label overlap and must be
recomputed within strata before use.

Reproduce: `python src/pilot/analyze.py --run results/pilot_qwen3_4b --acts data/activations/pilot_qwen3_4b --label "Qwen3-4B"`.

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
