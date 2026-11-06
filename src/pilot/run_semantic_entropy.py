"""Discrete semantic entropy (Farquhar et al., Nature 2024) on the pilot questions.

Samples K answers per question at temperature 1.0, clusters them by meaning, and
records the entropy over clusters. Short trivia answers let us cluster by normalized
string match (plus gold-alias match) instead of the paper's NLI entailment model; the
notebook says so where the numbers are shown.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from normalize import contains_match, normalize  # noqa: E402
from run_pilot import build_prompts, load_config, load_questions, pick_device  # noqa: E402

GOLD_CLUSTER = "<gold>"


def cluster_key(sample: str, aliases: list[str]) -> str:
    """Samples that match any gold alias share one cluster; others cluster by normalized text."""
    if contains_match(sample, aliases):
        return GOLD_CLUSTER
    return normalize(sample)


def discrete_semantic_entropy(samples: list[str], aliases: list[str]) -> dict:
    keys = [cluster_key(s, aliases) for s in samples]
    counts = Counter(keys)
    n = len(keys)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return {"semantic_entropy": entropy, "n_clusters": len(counts),
            "frac_gold": counts.get(GOLD_CLUSTER, 0) / n, "cluster_keys": keys}


@torch.no_grad()
def sample_batch(model, tok, prompts: list[str], k: int, max_new_tokens: int, device: str) -> list[list[str]]:
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                         top_k=0, num_return_sequences=k, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_questions", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--start", type=int, default=0, help="skip the first START questions (e.g. 500 for a held-out set)")
    ap.add_argument("--out_name", default="semantic_entropy.jsonl")
    args = ap.parse_args()
    if not 0 <= args.start < args.n_questions:
        ap.error(f"--start must be in [0, n_questions={args.n_questions})")
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = pick_device()
    out_path = Path(cfg["out_dir"]) / args.out_name
    # resumable: questions already in the output file are skipped and new ones appended
    done = {json.loads(line)["qid"] for line in out_path.open()} if out_path.exists() else set()

    print(f"device={device} model={cfg['model_id']} questions {args.start}..{args.n_questions - 1} k={args.k} "
          f"-> {out_path} ({len(done)} already done)", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.bfloat16).to(device).eval()
    rows = [r for r in load_questions({**cfg, "n_questions": args.n_questions})[args.start:] if r["qid"] not in done]
    prompts = build_prompts(tok, [r["question"] for r in rows])

    t0 = time.time()
    with out_path.open("a") as f:
        for i in tqdm(range(0, len(rows), args.batch_size), file=sys.stdout, mininterval=30):
            batch_rows = rows[i:i + args.batch_size]
            samples = sample_batch(model, tok, prompts[i:i + args.batch_size], args.k, cfg["max_new_tokens"], device)
            for r, s in zip(batch_rows, samples):
                rec = {"qid": r["qid"], "samples": s, **discrete_semantic_entropy(s, r["aliases"])}
                f.write(json.dumps(rec) + "\n")
            f.flush()
    print(f"wrote {out_path} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
