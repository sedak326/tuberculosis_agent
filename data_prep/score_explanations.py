#!/usr/bin/env python3
"""
score_explanations.py — Use GPT-4o to score model MCQ outputs.

For each question, a single GPT-4o call:
  1. Extracts the predicted answer letter from the model's raw output
  2. Scores the explanation on four dimensions (1-5) against the gold explanation

Output per record:
  predicted, is_correct, penalized_accuracy,
  factual_accuracy, coherence, naturalness, completeness

Default mode uses the Batch API (50% cheaper than synchronous calls, since
none of this needs to come back instantly) — submits all pending questions
as one batch job, polls until it completes (usually minutes, up to 24h by
OpenAI's own SLA), then scores the results. The batch id is saved alongside
--out so a rerun resumes polling an in-flight batch instead of resubmitting.

Usage:
    export OPENAI_API_KEY=sk-...
    python score_explanations.py \
        --results /uss/skavlak/tb_corpus_v3/eval_results/8b_sft.jsonl \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_sft_scored.jsonl

    # Resume a partial run (sync mode) or an in-flight/partially-processed batch:
    python score_explanations.py ... --resume

    # Old one-call-at-a-time behavior (useful for a quick few-item debug run):
    python score_explanations.py ... --sync
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

METRICS = ["factual_accuracy", "coherence", "naturalness", "completeness"]


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


def build_prompt(mcq: dict, result: dict, correct_letter: str, choices: dict) -> str:
    raw_output = result.get("raw_output", "").strip()
    return JUDGE_PROMPT.format(
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


def make_record(mcq_id: str, correct_letter: str, scores: dict) -> dict:
    predicted = scores.get("predicted")
    if isinstance(predicted, str):
        predicted = predicted.upper().strip()
        if predicted not in ("A", "B", "C", "D"):
            predicted = None

    is_correct = predicted == correct_letter if predicted else False
    pen_acc = penalized_accuracy(predicted, correct_letter)

    return {
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


def print_summary(records: list[dict]):
    scored = len(records)
    if scored == 0:
        return
    correct = sum(1 for r in records if r["is_correct"])
    no_answer = sum(1 for r in records if r["predicted"] is None)
    score_sum = sum(r["penalized_accuracy"] for r in records)
    print(f"\nDone. {scored} questions scored.")
    print(f"  Accuracy          : {correct}/{scored} = {correct/scored:.3f}")
    print(f"  Penalized accuracy: {score_sum:.1f}/{scored:.1f} = {score_sum/scored:.3f}")
    print(f"  No answer         : {no_answer}")
    for m in METRICS:
        avg = sum(r[m] or 0 for r in records) / scored
        print(f"  Avg {m:20s}: {avg:.2f} / 5")


def run_sync(client, args, to_score, mcqs, presented_choices, out_path):
    metrics = METRICS
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

            correct_letter = result["correct_presented"]
            choices = presented_choices[mcq_id]
            prompt = build_prompt(mcq, result, correct_letter, choices)

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
                record = make_record(mcq_id, correct_letter, scores)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if record["predicted"] is None:
                    no_answer += 1
                elif record["is_correct"]:
                    correct += 1
                score_sum += record["penalized_accuracy"]
                for m in metrics:
                    totals[m] += scores.get(m, 0)
                scored += 1

                if scored % 50 == 0:
                    print(f"  [{scored}/{len(to_score)}]  "
                          f"acc={correct/scored:.3f}  "
                          f"pen={score_sum/scored:.3f}  " +
                          "  ".join(f"{m[:4]}={totals[m]/scored:.2f}" for m in metrics))

                time.sleep(0.1)

            except Exception as e:
                print(f"  [error] {mcq_id}: {e}")
                time.sleep(2)

    if scored > 0:
        print_summary(load_jsonl_dedup(str(out_path)))
        print(f"Results saved to {out_path}")


def run_batch(client, args, to_score, mcqs, presented_choices, out_path):
    batch_id_path = out_path.with_name(out_path.name + ".batch_id")

    batch_id = None
    if batch_id_path.exists():
        batch_id = batch_id_path.read_text().strip()
        print(f"Found in-flight batch id {batch_id}, checking status...")

    if batch_id is None:
        if not to_score:
            print("Nothing to score.")
            return

        valid_to_score = []
        input_path = out_path.with_name(out_path.name + ".batch_input.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for result in to_score:
                mcq_id = result["id"]
                mcq = mcqs.get(mcq_id)
                if mcq is None:
                    print(f"  [skip] {mcq_id} not found in MCQ file")
                    continue
                correct_letter = result["correct_presented"]
                choices = presented_choices[mcq_id]
                prompt = build_prompt(mcq, result, correct_letter, choices)
                line = {
                    "custom_id": mcq_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 100,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                valid_to_score.append(mcq_id)

        print(f"Submitting batch of {len(valid_to_score)} requests...")
        uploaded = client.files.create(file=open(input_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id_path.write_text(batch.id)
        batch_id = batch.id
        print(f"Batch submitted: {batch_id}")

    # Poll until the batch reaches a terminal state.
    poll_interval = args.poll_interval
    terminal_ok = {"completed"}
    terminal_bad = {"failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  batch {batch_id}: status={batch.status}  "
              f"completed={counts.completed}/{counts.total}  failed={counts.failed}")
        if batch.status in terminal_ok:
            break
        if batch.status in terminal_bad:
            print(f"Batch ended with status={batch.status}. "
                  f"Delete {batch_id_path} to resubmit (check errors first).")
            return
        time.sleep(poll_interval)

    correct_letter_by_id = {}
    for result in to_score:
        correct_letter_by_id[result["id"]] = result["correct_presented"]

    scored_this_run = []
    with open(out_path, "a", encoding="utf-8") as f:
        output_content = client.files.content(batch.output_file_id)
        for line in output_content.text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            mcq_id = entry["custom_id"]
            correct_letter = correct_letter_by_id.get(mcq_id)
            if correct_letter is None:
                continue
            body = entry.get("response", {}).get("body")
            if entry.get("error") or body is None:
                print(f"  [error] {mcq_id}: {entry.get('error')}")
                continue
            try:
                scores = json.loads(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"  [error] {mcq_id}: malformed response ({e})")
                continue
            record = make_record(mcq_id, correct_letter, scores)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored_this_run.append(record)

        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            for line in error_content.text.splitlines():
                if line.strip():
                    print(f"  [batch error] {line}")

    batch_id_path.unlink(missing_ok=True)
    input_path = out_path.with_name(out_path.name + ".batch_input.jsonl")
    input_path.unlink(missing_ok=True)

    print(f"\n{len(scored_this_run)} questions scored this run.")
    print_summary(load_jsonl_dedup(str(out_path)))
    print(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq", default="/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync", action="store_true",
                         help="Fall back to the old one-call-at-a-time behavior "
                              "instead of the (default, 50%% cheaper) Batch API.")
    parser.add_argument("--poll-interval", type=int, default=30,
                         help="Seconds between batch status checks (batch mode only).")
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
    print(f"Scoring {len(to_score)} / {len(results)} questions with {args.model} "
          f"({'sync' if args.sync else 'batch'} mode)")

    if args.sync:
        run_sync(client, args, to_score, mcqs, presented_choices, out_path)
    else:
        run_batch(client, args, to_score, mcqs, presented_choices, out_path)


if __name__ == "__main__":
    main()
