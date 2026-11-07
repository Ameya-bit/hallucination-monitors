"""Load one pilot run's answers with judge labels, confidence and semantic-entropy splits.

Same loading rules as notebooks/replication.ipynb: answers the judge called ambiguous are dropped,
"confident" means first-token probability >= THRESHOLD, and the hard set is confident answers whose
5 resamples all agree (semantic entropy 0). Questions 1-500 are the "explore" split (used to choose
layers) and 501-2000 the "holdout" split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.8
POSITION_FILES = {"last_prompt": "last_prompt_token.npy", "first_answer": "first_answer_token.npy"}
SE_FILES = (("semantic_entropy.jsonl", "explore"), ("semantic_entropy_holdout.jsonl", "holdout"))


def pair_key(qid: pd.Series, answer: pd.Series) -> pd.Series:
    return qid + "\t" + answer.str.strip().str.lower()


def load_run(root: Path, run: str) -> pd.DataFrame:
    run_dir = root / "results" / run
    d = pd.read_json(run_dir / "records.jsonl", lines=True)
    d["row"] = np.arange(len(d))  # row in the activation arrays
    j = pd.read_json(run_dir / "judge_labels.jsonl", lines=True)
    verdict = j.assign(key=pair_key(j.qid, j.answer)).drop_duplicates("key").set_index("key").verdict
    d["verdict"] = pair_key(d.qid, d.answer).map(verdict)
    if d.verdict.isna().any():
        raise ValueError(f"{run}: answers without a judge verdict")
    d = d[d.verdict != "ambiguous"].reset_index(drop=True)
    d["correct"] = (d.verdict == "correct").astype(int)
    d["confident"] = d.first_tok_prob >= THRESHOLD
    parts = [pd.read_json(p, lines=True)[["qid", "semantic_entropy"]].assign(se_split=split)
             for name, split in SE_FILES if (p := run_dir / name).exists()]
    se = pd.concat(parts, ignore_index=True)
    if not se.qid.is_unique:
        raise ValueError(f"{run}: a question appears in both semantic-entropy files")
    d = d.merge(se, on="qid", how="left")
    d["hard"] = d.confident & (d.semantic_entropy == 0)
    d["hard_explore"] = d.hard & (d.se_split == "explore")
    d["hard_holdout"] = d.hard & (d.se_split == "holdout")
    return d


def load_acts(root: Path, run: str, rows: np.ndarray, position: str = "last_prompt") -> np.ndarray:
    """Activations (len(rows), n_layers + 1, d_model) at one token position, memory-mapped."""
    if position not in POSITION_FILES:
        raise ValueError(f"position must be one of {sorted(POSITION_FILES)}, got {position!r}")
    return np.load(root / "data" / "activations" / run / POSITION_FILES[position], mmap_mode="r")[rows]
