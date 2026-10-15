#!/usr/bin/env python3
"""
pairwise_judge.py — GPT-4o picks its preferred answer between two model outputs for the
same question, instead of scoring each independently. Complementary to
score_explanations.py's absolute 1-5 rubric: pairwise judging is typically more sensitive
when both models already score near the ceiling on absolute scales.

Controls for judge position bias: which answer is shown first is randomized per-item
(fixed seed), and the model never sees which system (base/fine-tuned) produced which
answer - only "Response 1" / "Response 2".

Usage (eyeball-comparison mode, the default - reads base_output/run4_output pairs
already generated for the eyeball test):
    python pairwise_judge.py \
        --pairs eyeball_comparison.json \
        --a-key base_output --b-key run4_output \
        --a-name base --b-name run4_sft_instruct \
        --out /uss/skavlak/version_3/eval_results/pairwise_eyeball.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """\
You are an expert in tuberculosis (TB) research, acting as an impartial judge comparing \
two candidate answers to the same researcher question. LLM judges are well documented to \
systematically over-reward longer, more exhaustive-sounding responses regardless of \
whether the extra length is actually substantive - you must actively correct for this, \
not just avoid mentioning length."""

JUDGE_PROMPT = """\
Question / scenario:
{question}

Response 1 ({len1} words):
{resp1}

Response 2 ({len2} words):
{resp2}

These may differ substantially in length. Before judging, explicitly work through: for \
whichever response is longer, is the EXTRA content (beyond what the shorter response \
covers) actually correct, specific, and load-bearing for answering this question - or is \
it generic, hedging, redundant, or padding? A focused response that correctly nails the \
one or two things that actually matter is better than a longer response that pads out \
with generic/redundant material. Conversely, a longer response is genuinely better only \
if its extra content adds real, correct, non-obvious substance the shorter one is missing.

Judge on: factual accuracy, correctness and depth of reasoning, whether it actually \
engages with the specifics of this scenario, and usefulness to the researcher. If they are \
genuinely comparable once you've accounted for padding, say "tie" rather than forcing a pick.

Respond with valid JSON only, using this exact format:
{{
  "length_check": "1 sentence: is the longer response's extra content substantive or padding?",
  "verdict": "1" or "2" or "tie",
  "reasoning": "1-3 sentences on what specifically made the difference (or why they're comparable)"
}}"""


def parse_response(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_pairs(path: str, a_key: str, b_key: str, id_key: str, q_key: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8")) if path.endswith(".json") else \
        [json.loads(l) for l in open(path, encoding="utf-8")]
    out = []
    for i, r in enumerate(data):
        a = r.get(a_key)
        b = r.get(b_key)
        if not a or not b:
            continue
        out.append({
            "id": r.get(id_key, f"item_{i:03d}"),
            "category": r.get("category", ""),
            "question": r.get(q_key, r.get("prompt", r.get("question", ""))),
            "a": a, "b": b,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--a-key", default="base_output")
    ap.add_argument("--b-key", default="run4_output")
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--id-key", default="id")
    ap.add_argument("--q-key", default="prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rng = random.Random(args.seed)

    items = load_pairs(args.pairs, args.a_key, args.b_key, args.id_key, args.q_key)
    print(f"Loaded {len(items)} pairs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["id"])
        print(f"Resuming: {len(done)} already judged")

    wins = {args.a_name: 0, args.b_name: 0, "tie": 0}
    n = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for item in items:
            if item["id"] in done:
                continue
            a_first = rng.random() < 0.5
            resp1, resp2 = (item["a"], item["b"]) if a_first else (item["b"], item["a"])
            label1, label2 = (args.a_name, args.b_name) if a_first else (args.b_name, args.a_name)
            len1, len2 = len(resp1.split()), len(resp2.split())

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": JUDGE_PROMPT.format(
                            question=item["question"], resp1=resp1, resp2=resp2,
                            len1=len1, len2=len2)},
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                parsed = parse_response(resp.choices[0].message.content)
                if parsed is None or parsed.get("verdict") not in {"1", "2", "tie"}:
                    print(f"  [skip] bad response for {item['id']}")
                    continue

                winner = "tie" if parsed["verdict"] == "tie" else (label1 if parsed["verdict"] == "1" else label2)
                wins[winner] += 1
                n += 1
                record = {
                    "id": item["id"], "category": item["category"],
                    "a_shown_first": a_first, "winner": winner,
                    "length_check": parsed.get("length_check", ""),
                    "reasoning": parsed.get("reasoning", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{n}] {item['id']} -> {winner}")
                time.sleep(0.3)

            except Exception as e:
                print(f"  [error] {e}")
                time.sleep(2)

    print(f"\nDone. {n} judged.")
    print(f"Wins: {args.a_name}={wins[args.a_name]}  {args.b_name}={wins[args.b_name]}  tie={wins['tie']}")


if __name__ == "__main__":
    main()
