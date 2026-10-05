#!/usr/bin/env python3
"""
extract_predictions.py — Extract predicted answer letters from raw model outputs.

Reads evaluate_emcqa.py output, parses the predicted letter from raw_output using
regex patterns, recomputes is_correct and penalized_accuracy, writes corrected JSONL.

Usage:
    python extract_predictions.py \
        --results /uss/skavlak/tb_corpus_v3/eval_results/8b_sft.jsonl \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_sft_parsed.jsonl

    # Process all conditions at once:
    python extract_predictions.py --all
"""

import argparse
import json
import re
from pathlib import Path

RESULTS_DIR = Path("/uss/skavlak/tb_corpus_v3/eval_results")

ALL_CONDITIONS = [
    "8b_base",
    "8b_sft",
    "8b_sft_miwv",
    "8b_cpt_sft",
    "8b_cpt_sft_miwv",
]

PATTERNS = [
    r"Answer:\s*([A-D])\b",                                      # Answer: B
    r"answer\s+is\s+(?:option\s+)?([A-D])\b",                   # answer is B / answer is option B
    r"correct\s+answer\s+is\s+(?:option\s+)?([A-D])\b",         # correct answer is B / correct answer is option B
    r"\bchoose\s+(?:option\s+)?([A-D])\b",                      # choose B / choose option B
    r"\bselect\s+(?:option\s+)?([A-D])\b",                      # select B
    r"^([A-D])[).]\s",                                           # B) or B. at start of line
    r"\b([A-D])\s+is\s+(?:the\s+)?correct\b",                   # B is correct / B is the correct
    r"option\s+([A-D])\s+is\s+(?:the\s+)?correct\b",            # option B is correct
    r"^\s*([A-D])\s*$",                                          # bare letter on its own line
]


def extract_letter(raw_output: str) -> str | None:
    for pattern in PATTERNS:
        m = re.search(pattern, raw_output, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def penalized_accuracy(predicted: str | None, correct: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted == correct else -0.5


def process(results_path: Path, out_path: Path):
    # Deduplicate by id (last write wins — handles sft_miwv double-write)
    rows = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["id"]] = obj

    total = len(rows)
    correct = 0
    no_answer = 0
    score_sum = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows.values():
            predicted = extract_letter(row["raw_output"])
            correct_letter = row["correct_presented"]
            is_correct = predicted == correct_letter if predicted else False
            pen_acc = penalized_accuracy(predicted, correct_letter)

            row["predicted"] = predicted
            row["is_correct"] = is_correct
            row["penalized_accuracy"] = pen_acc

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if predicted is None:
                no_answer += 1
            elif is_correct:
                correct += 1
            score_sum += pen_acc

    print(f"{results_path.name}")
    print(f"  n={total}  correct={correct}  no_answer={no_answer}  wrong={total-correct-no_answer}")
    print(f"  Accuracy          : {correct/total:.3f}")
    print(f"  Penalized accuracy: {score_sum:.1f}/{total:.1f} = {score_sum/total:.3f}")
    print(f"  -> {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", help="Path to single eval JSONL")
    parser.add_argument("--out", help="Output path (required with --results)")
    parser.add_argument("--all", action="store_true", help="Process all conditions")
    args = parser.parse_args()

    if args.all:
        for name in ALL_CONDITIONS:
            src = RESULTS_DIR / f"{name}.jsonl"
            dst = RESULTS_DIR / f"{name}_parsed.jsonl"
            if not src.exists():
                print(f"  [skip] {src} not found")
                continue
            process(src, dst)
    elif args.results:
        out = Path(args.out) if args.out else Path(args.results).with_suffix("").with_name(
            Path(args.results).stem + "_parsed.jsonl"
        )
        process(Path(args.results), out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
