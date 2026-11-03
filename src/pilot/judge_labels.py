"""LLM-judge correctness labels (Claude Haiku 5.5) for greedy answers and SE samples.

Lexical contains-match over-credits and under-credits short answers (Janiak et al., EMNLP 2025),
so every unique (question, answer) pair in a run is judged against the gold aliases. Output is a
separate file, `results/<run>/judge_labels.jsonl`; records.jsonl is never modified. The script is
resumable: pairs already in the output file are skipped.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_pilot import load_config, load_questions  # noqa: E402

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2048
MAX_ALIASES = 25
log = logging.getLogger("judge")

SYSTEM = """You grade answers to closed-book trivia questions.

You get a question, the reference answer with its accepted aliases, and a model's short answer.
Decide whether the model's answer refers to the same entity, value, or fact as the reference.

- correct: same referent. Spelling variants, missing or extra first names, titles, articles,
  abbreviations, and equivalent units or number formats count as correct.
- incorrect: a different referent, a broader or narrower one that would not be accepted as the
  answer (e.g. a country when a city is asked), or no answer at all.
- ambiguous: the question or reference is itself flawed (outdated, contradictory, or the
  question admits several valid answers and the model gave a defensible one not in the list),
  so a fair grader could not decide.

Judge the reference, not your own knowledge, unless the reference is clearly wrong; in that case
say ambiguous. Keep the reason to one short sentence."""


class Verdict(BaseModel):
    verdict: Literal["correct", "incorrect", "ambiguous"]
    reason: str


def pair_key(qid: str, answer: str) -> str:
    return f"{qid}\t{answer.strip().lower()}"


def collect_pairs(run_dir: Path) -> list[tuple[str, str]]:
    """Unique (qid, answer) pairs from the greedy records and the semantic-entropy samples."""
    seen: dict[str, tuple[str, str]] = {}
    for line in (run_dir / "records.jsonl").open():
        r = json.loads(line)
        seen.setdefault(pair_key(r["qid"], r["answer"]), (r["qid"], r["answer"].strip()))
    se_path = run_dir / "semantic_entropy.jsonl"
    if se_path.exists():
        for line in se_path.open():
            s = json.loads(line)
            for ans in s["samples"]:
                seen.setdefault(pair_key(s["qid"], ans), (s["qid"], ans.strip()))
    return list(seen.values())


def build_prompt(question: str, gold: str, aliases: list[str], answer: str) -> str:
    uniq = list(dict.fromkeys(a for a in aliases if a))[:MAX_ALIASES]
    return (f"Question: {question}\n"
            f"Reference answer: {gold}\n"
            f"Accepted aliases: {'; '.join(uniq)}\n"
            f"Model answer: {answer if answer else '(empty)'}")


async def judge_one(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, prompt: str) -> dict:
    async with sem:
        try:
            resp = await client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "low"},
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=Verdict,
            )
        except anthropic.APIStatusError as e:
            log.warning("API error %s: %s", e.status_code, e.message)
            return {"verdict": "error", "reason": f"api_{e.status_code}"}
        except anthropic.APIConnectionError as e:
            log.warning("connection error: %s", e)
            return {"verdict": "error", "reason": "connection"}
    if resp.stop_reason != "end_turn" or resp.parsed_output is None:
        return {"verdict": "error", "reason": f"stop_{resp.stop_reason}"}
    return resp.parsed_output.model_dump()


async def run(cfg: dict, limit: int | None, concurrency: int) -> None:
    run_dir = Path(cfg["out_dir"])
    out_path = run_dir / "judge_labels.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            if r["verdict"] != "error":
                done.add(pair_key(r["qid"], r["answer"]))

    questions = {q["qid"]: q for q in load_questions(cfg)}
    todo = [p for p in collect_pairs(run_dir) if pair_key(*p) not in done]
    if limit is not None:
        todo = todo[:limit]
    log.info("%s: %d pairs to judge (%d already done)", run_dir.name, len(todo), len(done))

    sem = asyncio.Semaphore(concurrency)
    async with anthropic.AsyncAnthropic(max_retries=6) as client:
        async def task(qid: str, answer: str) -> dict:
            q = questions[qid]
            res = await judge_one(client, sem, build_prompt(q["question"], q["gold"], q["aliases"], answer))
            return {"qid": qid, "answer": answer, **res, "judge": MODEL}

        with out_path.open("a") as f:
            for fut in tqdm.as_completed([task(*p) for p in todo], total=len(todo), mininterval=10):
                f.write(json.dumps(await fut) + "\n")
                f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pending pairs")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    asyncio.run(run(load_config(args.config), args.limit, args.concurrency))


if __name__ == "__main__":
    main()
