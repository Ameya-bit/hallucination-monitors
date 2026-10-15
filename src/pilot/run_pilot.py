"""Gate A pilot: greedy closed-book TriviaQA on a small model.

For each question, records the generated answer, first-answer-token probability,
top-2 gap, sequence log-prob, correctness, and residual-stream activations at every
layer for (a) the last prompt token and (b) the first answer token.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from normalize import contains_match, exact_match  # noqa: E402

SYSTEM = "You are a trivia expert. Answer with only the answer, no explanation."


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_questions(cfg: dict) -> list[dict]:
    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=cfg["split"], streaming=True)
    rows = []
    for ex in ds:
        aliases = list(ex["answer"]["aliases"]) + list(ex["answer"]["normalized_aliases"])
        rows.append({"qid": ex["question_id"], "question": ex["question"], "aliases": aliases,
                     "gold": ex["answer"]["value"]})
        if len(rows) >= cfg["n_questions"]:
            break
    return rows


def build_prompts(tok, questions: list[str]) -> list[str]:
    prompts = []
    for q in questions:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    return prompts


def stack_layers(hidden_states_at_step, position: int) -> np.ndarray:
    """hidden_states_at_step: tuple(len=n_layers+1) of [B, T, D]. Returns [B, n_layers+1, D] fp16."""
    return torch.stack([h[:, position, :] for h in hidden_states_at_step], dim=1).to(torch.float16).cpu().numpy()


@torch.no_grad()
def run_batch(model, tok, prompts: list[str], max_new_tokens: int, device: str):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         output_scores=True, output_hidden_states=True, return_dict_in_generate=True,
                         pad_token_id=tok.pad_token_id)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    # scores: tuple(len=steps) of [B, V] logits for each generated token
    probs0 = torch.softmax(out.scores[0].float(), dim=-1)
    top2 = probs0.topk(2, dim=-1).values
    first_tok_prob = probs0.gather(1, gen_ids[:, :1]).squeeze(1)
    top2_gap = top2[:, 0] - top2[:, 1]
    seq_logprob = torch.zeros(gen_ids.shape[0], device=device)
    min_tok_prob = torch.ones(gen_ids.shape[0], device=device)
    for step, logits in enumerate(out.scores):
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_ids = gen_ids[:, step]
        alive = (tok_ids != tok.pad_token_id) & (tok_ids != tok.eos_token_id)
        step_lp = lp.gather(1, tok_ids[:, None]).squeeze(1)
        seq_logprob += torch.where(alive, step_lp, torch.zeros_like(step_lp))
        min_tok_prob = torch.where(alive, torch.minimum(min_tok_prob, step_lp.exp()), min_tok_prob)
    # hidden_states[0]: prompt pass (all positions). hidden_states[1]: pass over first generated token.
    act_last_prompt = stack_layers(out.hidden_states[0], -1)
    act_first_answer = stack_layers(out.hidden_states[1], -1) if len(out.hidden_states) > 1 else act_last_prompt
    answers = tok.batch_decode(gen_ids, skip_special_tokens=True)
    return {
        "answers": [a.strip() for a in answers],
        "first_tok_prob": first_tok_prob.cpu().numpy(),
        "top2_gap": top2_gap.cpu().numpy(),
        "seq_logprob": seq_logprob.cpu().numpy(),
        "min_tok_prob": min_tok_prob.cpu().numpy(),
        "act_last_prompt": act_last_prompt,
        "act_first_answer": act_first_answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = pick_device()
    out_dir, act_dir = Path(cfg["out_dir"]), Path(cfg["act_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} model={cfg['model_id']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.bfloat16).to(device).eval()

    rows = load_questions(cfg)
    print(f"loaded {len(rows)} questions", flush=True)
    prompts = build_prompts(tok, [r["question"] for r in rows])

    records, acts_lp, acts_fa = [], [], []
    bs = cfg["batch_size"]
    t0 = time.time()
    for i in tqdm(range(0, len(rows), bs), file=sys.stdout, mininterval=30):
        batch_rows, batch_prompts = rows[i:i + bs], prompts[i:i + bs]
        res = run_batch(model, tok, batch_prompts, cfg["max_new_tokens"], device)
        for j, r in enumerate(batch_rows):
            ans = res["answers"][j]
            records.append({
                **{k: r[k] for k in ("qid", "question", "gold")},
                "answer": ans,
                "exact_match": exact_match(ans, r["aliases"]),
                "contains_match": contains_match(ans, r["aliases"]),
                "first_tok_prob": float(res["first_tok_prob"][j]),
                "top2_gap": float(res["top2_gap"][j]),
                "seq_logprob": float(res["seq_logprob"][j]),
                "min_tok_prob": float(res["min_tok_prob"][j]),
            })
        acts_lp.append(res["act_last_prompt"])
        acts_fa.append(res["act_first_answer"])
        if (i // bs) % 25 == 0:
            with open(out_dir / "records.jsonl", "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

    with open(out_dir / "records.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    np.save(act_dir / "last_prompt_token.npy", np.concatenate(acts_lp))
    np.save(act_dir / "first_answer_token.npy", np.concatenate(acts_fa))

    n = len(records)
    correct = sum(r["contains_match"] for r in records)
    conf = [r for r in records if r["first_tok_prob"] >= cfg["confident_threshold"]]
    conf_wrong = [r for r in conf if not r["contains_match"]]
    summary = {
        "n": n, "accuracy_contains": correct / n,
        "accuracy_exact": sum(r["exact_match"] for r in records) / n,
        "n_confident": len(conf), "n_confident_wrong": len(conf_wrong),
        "confident_threshold": cfg["confident_threshold"],
        "gate_a_pass_150": len(conf_wrong) >= 150,
        "minutes": (time.time() - t0) / 60,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
