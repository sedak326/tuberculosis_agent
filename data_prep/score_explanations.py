#!/usr/bin/env python3
"""
score_explanations.py — Use GPT-4o to score model MCQ outputs.

For each question, a single GPT-4o call:
  1. Extracts the predicted answer letter from the model's raw output
  2. Scores the explanation on four dimensions (1-5) against the gold explanation

Output per record:
  predicted, is_correct, penalized_accuracy,
  factual_accuracy, coherence, naturalness, completeness

Usage:
    export OPENAI_API_KEY=sk-...
    python score_explanations.py \
        --results /uss/skavlak/tb_corpus_v3/eval_results/8b_sft.jsonl \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_sft_scored.jsonl

    # Resume a partial run:
    python score_explanations.py ... --resume
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

JUDGE_SYSTEM = """\
You are an expert evaluator of tuberculosis research knowledge. \
You extract answers and score explanations from model responses to multiple-choice questions."""

JUDGE_PROMPT = """\
Question: {question}

Answer choices:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Correct answer: {correct_letter}) {correct_text}

Gold explanation (written by an expert):
{gold_explanation}

Model response:
{raw_output}

Tasks:
1. Identify which of the four answer choices (A, B, C, or D) the model's response supports. \
Do NOT require an explicit letter statement — the model may never say "Answer: B" and instead \
just explain the underlying fact or claim directly (e.g. stating a specific number, mechanism, \
or finding). Read the substance of the response and match it against the text of the four answer \
choices above: if the content of the response clearly corresponds to one choice's text \
(even paraphrased), that is the predicted letter. Only use null if the response genuinely does \
not support any single one of the four choices over the others (e.g. it's off-topic, refuses, \
or is too vague to distinguish between choices).
2. Score the model's explanation on four dimensions (each 1–5):
   - factual_accuracy: Are the TB biology facts stated in the explanation correct? \
(1 = major errors, 5 = fully accurate)
   - coherence: Is the explanation logically structured and internally consistent? \
(1 = contradictory or disorganised, 5 = clear and consistent)
   - naturalness: Is it fluent and reads like a domain expert wrote it? \
(1 = unnatural or awkward, 5 = professional and fluent)
   - completeness: Does it sufficiently explain why the correct answer is right? \
(1 = missing key reasoning, 5 = thorough and complete)

Respond with JSON only — no other text:
{{"predicted": "A" or null, "factual_accuracy": 1-5, "coherence": 1-5, "naturalness": 1-5, "completeness": 1-5}}"""


def shuffle_choices(mcq: dict, rng: "random.Random") -> tuple[dict, str]:
    """Mirrors evaluate_emcqa.py's shuffle_choices exactly, so the presented
    choices/letters can be deterministically reconstructed from the MCQ file
    + seed alone (the raw eval output never saved the shuffled choices)."""
    letters = ["A", "B", "C", "D"]
    original_correct_text = mcq["choices"][mcq["correct"]]
    texts = [mcq["choices"][l] for l in letters]
    rng.shuffle(texts)
    shuffled_choices = dict(zip(letters, texts))
    new_correct = next(l for l, t in shuffled_choices.items() if t == original_correct_text)
    return shuffled_choices, new_correct


def build_presented_choices(mcqs_in_order: list[dict], seed: int) -> dict:
    """Returns {mcq_id: {letter: text}} using the presented (possibly shuffled)
    choices, replaying the exact same RNG sequence evaluate_emcqa.py used."""
    rng = random.Random(seed)
    presented = {}
    for m in mcqs_in_order:
        choices, _ = shuffle_choices(m, rng)
        presented[m["id"]] = choices
    return presented


def penalized_accuracy(predicted: str | None, correct: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted == correct else -0.5


def load_jsonl_dedup(path: str) -> list[dict]:
    """Load JSONL deduplicating by id (last write wins)."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["id"]] = obj
    return list(rows.values())


def load_mcqs(path: str) -> dict:
    mcqs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mcqs[obj["id"]] = obj
    return mcqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq", default="/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--randomize-answers", action="store_true",
                         help="Must match whether the eval run that produced --results used "
                              "evaluate_emcqa.py's --randomize-answers, so the presented choice "
                              "text can be correctly reconstructed for the judge prompt.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the --seed used by evaluate_emcqa.py for this run.")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    mcqs = load_mcqs(args.mcq)
    results = load_jsonl_dedup(args.results)

    if args.randomize_answers:
        # mcqs dict preserves insertion order (file order) in Python 3.7+, which
        # is required to replay the exact same RNG sequence evaluate_emcqa.py used.
        presented_choices = build_presented_choices(list(mcqs.values()), args.seed)
    else:
        presented_choices = {mid: m["choices"] for mid, m in mcqs.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already scored")

    to_score = [r for r in results if r["id"] not in done_ids]
    print(f"Scoring {len(to_score)} / {len(results)} questions with {args.model}")

    metrics = ["factual_accuracy", "coherence", "naturalness", "completeness"]
    totals = {m: 0 for m in metrics}
    correct = 0
    score_sum = 0.0
    no_answer = 0
    scored = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for result in to_score:
            mcq_id = result["id"]
            mcq = mcqs.get(mcq_id)
            if mcq is None:
                print(f"  [skip] {mcq_id} not found in MCQ file")
                continue

            raw_output = result.get("raw_output", "").strip()
            correct_letter = result["correct_presented"]
            choices = presented_choices[mcq_id]

            prompt = JUDGE_PROMPT.format(
                question=mcq["question"],
                choice_a=choices["A"],
                choice_b=choices["B"],
                choice_c=choices["C"],
                choice_d=choices["D"],
                correct_letter=correct_letter,
                correct_text=mcq["choices"][mcq["correct"]],
                gold_explanation=mcq["explanation"],
                raw_output=raw_output or "(no response)",
            )

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                )
                scores = json.loads(response.choices[0].message.content)

                predicted = scores.get("predicted")
                if isinstance(predicted, str):
                    predicted = predicted.upper().strip()
                    if predicted not in ("A", "B", "C", "D"):
                        predicted = None

                is_correct = predicted == correct_letter if predicted else False
                pen_acc = penalized_accuracy(predicted, correct_letter)

                record = {
                    "id": mcq_id,
                    "predicted": predicted,
                    "correct_presented": correct_letter,
                    "is_correct": is_correct,
                    "penalized_accuracy": pen_acc,
                    "factual_accuracy": scores.get("factual_accuracy"),
                    "coherence": scores.get("coherence"),
                    "naturalness": scores.get("naturalness"),
                    "completeness": scores.get("completeness"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if predicted is None:
                    no_answer += 1
                elif is_correct:
                    correct += 1
                score_sum += pen_acc
                for m in metrics:
                    totals[m] += scores.get(m, 0)
                scored += 1

                if scored % 50 == 0:
                    total_so_far = scored
                    print(f"  [{scored}/{len(to_score)}]  "
                          f"acc={correct/total_so_far:.3f}  "
                          f"pen={score_sum/total_so_far:.3f}  " +
                          "  ".join(f"{m[:4]}={totals[m]/total_so_far:.2f}" for m in metrics))

                time.sleep(0.1)

            except Exception as e:
                print(f"  [error] {mcq_id}: {e}")
                time.sleep(2)

    if scored > 0:
        print(f"\nDone. {scored} questions scored.")
        print(f"  Accuracy          : {correct}/{scored} = {correct/scored:.3f}")
        print(f"  Penalized accuracy: {score_sum:.1f}/{scored:.1f} = {score_sum/scored:.3f}")
        print(f"  No answer         : {no_answer}")
        for m in metrics:
            print(f"  Avg {m:20s}: {totals[m]/scored:.2f} / 5")
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
