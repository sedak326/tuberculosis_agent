#!/usr/bin/env python3
"""
generate_mcq.py — Generate E-MCQA dataset from held-out eval corpus using GPT-4.

Each MCQ includes:
  - question requiring TB domain knowledge
  - 4 choices (one correct, three plausible distractors)
  - correct answer letter
  - gold explanation (why correct is right, why distractors are wrong)

Usage:
    python generate_mcq.py \
        --corpus /uss/skavlak/tb_corpus_v3/corpus_v3_eval.jsonl \
        --out    /uss/skavlak/tb_corpus_v3/mcq_eval.jsonl \
        --n      800
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
You are an expert in tuberculosis (TB) research with deep knowledge of microbiology, \
molecular biology, genomics, drug resistance, host-pathogen interactions, and TB therapeutics. \
Your task is to write high-quality multiple-choice questions to evaluate a large language model \
that has been fine-tuned to serve as a TB research assistant. The questions should rigorously \
test whether the model has acquired genuine TB domain knowledge — not just surface-level pattern \
matching — across a range of cognitive skills."""

MCQ_PROMPT = """\
You are building an evaluation dataset to test a fine-tuned TB research assistant LLM. \
Using the passage below as a source of factual grounding, write one multiple-choice question \
that tests whether the model has genuinely learned TB domain knowledge.

The question must be grounded in the concepts from the passage but should test the model's \
understanding at one of the following Bloom's Taxonomy cognitive levels:

- "recall": retrieving a specific fact or definition from TB biology
- "understanding": explaining a concept or mechanism in the model's own words
- "application": applying TB knowledge to a specific experimental or clinical scenario
- "analysis": interpreting data, comparing mechanisms, or identifying relationships between concepts

The final dataset must contain exactly equal numbers of questions at each cognitive level. \
You will be told which levels still need questions — only generate a question at one of those levels. \
Current needed levels: {needed_levels}

The question should:
- Use the passage as factual grounding but NOT be answerable by simply matching words from it
- Require a TB research assistant to reason about mechanisms, not just retrieve text
- Have exactly one clearly correct answer
- Have three plausible distractors based on common misconceptions or related-but-incorrect facts
- Be accompanied by a detailed explanation of why the correct answer is right and each distractor is wrong

Passage:
{passage}

Also assign a topic tag for the primary TB research area covered (choose one):
"drug_resistance", "host_pathogen", "virulence", "genomics", "transcriptomics",
"metabolomics", "protein_structure", "drug_targets", "signal_transduction",
"epigenomics", "biofilm", "latency_dormancy", "vaccine", "diagnosis", "clinical"

Respond with valid JSON only, using this exact format:
{{
  "question": "...",
  "choices": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct": "A",
  "explanation": "The correct answer is A because ... Choice B is wrong because ... Choice C is wrong because ... Choice D is wrong because ...",
  "cognitive_level": "recall",
  "topic": "drug_resistance"
}}"""


def load_chunks(corpus_path: str, n: int, seed: int) -> list[dict]:
    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("content_type") != "text":
                continue
            text = obj.get("text", "").strip()
            if len(text) < 300:
                continue
            chunks.append(obj)

    random.seed(seed)
    random.shuffle(chunks)
    return chunks[:n]


def parse_response(text: str) -> dict | None:
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


VALID_LEVELS = {"recall", "understanding", "application", "analysis"}
VALID_TOPICS = {
    "drug_resistance", "host_pathogen", "virulence", "genomics", "transcriptomics",
    "metabolomics", "protein_structure", "drug_targets", "signal_transduction",
    "epigenomics", "biofilm", "latency_dormancy", "vaccine", "diagnosis", "clinical"
}


def validate_mcq(mcq: dict) -> bool:
    required = {"question", "choices", "correct", "explanation", "cognitive_level", "topic"}
    if not required.issubset(mcq.keys()):
        return False
    if set(mcq["choices"].keys()) != {"A", "B", "C", "D"}:
        return False
    if mcq["correct"] not in {"A", "B", "C", "D"}:
        return False
    if len(mcq["question"].strip()) < 20:
        return False
    if mcq["cognitive_level"] not in VALID_LEVELS:
        return False
    if mcq["topic"] not in VALID_TOPICS:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/uss/skavlak/tb_corpus_v3/corpus_v3_eval.jsonl")
    parser.add_argument("--out", default="/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl")
    parser.add_argument("--n", type=int, default=800, help="Number of MCQs to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated MCQs")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load already-generated if resuming
    done = set()
    if args.resume and Path(args.out).exists():
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj.get("source_chunk_id", ""))
        print(f"Resuming: {len(done)} MCQs already generated")

    chunks = load_chunks(args.corpus, args.n * 2, args.seed)  # oversample in case of failures
    print(f"Loaded {len(chunks)} candidate chunks from eval corpus")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_level = args.n // len(VALID_LEVELS)   # e.g. 800 // 4 = 200
    level_counts = {l: 0 for l in VALID_LEVELS}

    # Populate counts from already-done MCQs when resuming
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                lv = obj.get("cognitive_level")
                if lv in level_counts:
                    level_counts[lv] += 1

    generated = sum(level_counts.values())
    failed = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            needed = [l for l, c in level_counts.items() if c < per_level]
            if not needed or generated >= args.n:
                break

            chunk_id = chunk.get("chunk_id", chunk.get("document_name", "") + str(chunk.get("chunk_index", "")))
            if chunk_id in done:
                continue

            passage = chunk.get("text", "")[:2000]
            needed_str = ", ".join(f'"{l}"' for l in sorted(needed))

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": MCQ_PROMPT.format(
                            passage=passage, needed_levels=needed_str)},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                mcq = parse_response(raw)

                if mcq is None or not validate_mcq(mcq):
                    print(f"  [skip] invalid MCQ for chunk {chunk_id}")
                    failed += 1
                    continue

                level = mcq["cognitive_level"]
                if level_counts[level] >= per_level:
                    # model returned a full level — skip without penalty
                    continue

                record = {
                    "id": f"mcq_{generated:04d}",
                    "source_chunk_id": chunk_id,
                    "document_name": chunk.get("document_name", ""),
                    **mcq,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                level_counts[level] += 1
                generated += 1

                if generated % 50 == 0:
                    print(f"  Generated {generated}/{args.n}  {dict(level_counts)}")

                time.sleep(0.5)

            except Exception as e:
                print(f"  [error] {e}")
                failed += 1
                time.sleep(2)

    print(f"\nDone. {generated} MCQs saved to {out_path}  ({failed} failed/skipped)")
    print(f"Final level counts: {dict(level_counts)}")


if __name__ == "__main__":
    main()
