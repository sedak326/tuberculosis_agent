#!/usr/bin/env python3
"""Dump a human-readable sample of the SFT training data, grouped by category,
so it can be read through directly instead of scrolling raw JSONL."""
import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/uss/skavlak/tb_corpus_v3/training_data_v3_run2.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_cat[r["category"]].append(r)

    with open(args.out, "w", encoding="utf-8") as out:
        for cat in sorted(by_cat):
            examples = by_cat[cat]
            sample = rng.sample(examples, min(args.per_category, len(examples)))
            out.write(f"{'=' * 80}\nCATEGORY: {cat}  ({len(examples)} total in dataset)\n{'=' * 80}\n\n")
            for i, ex in enumerate(sample, 1):
                user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
                asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
                out.write(f"--- example {i} (source: {ex.get('source_chunk_id', '?')}) ---\n")
                out.write(f"USER:\n{user}\n\n")
                out.write(f"ASSISTANT:\n{asst}\n\n\n")

    print(f"Wrote sample to {args.out}")


if __name__ == "__main__":
    main()
