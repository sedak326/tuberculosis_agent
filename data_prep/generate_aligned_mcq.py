#!/usr/bin/env python3
"""
generate_aligned_mcq.py — Build an MCQ-shaped eval set whose QUESTION STYLE matches the
SFT training data (open-ended analysis questions), not the traditional recall-style
questions in mcq_eval.jsonl. Same held-out corpus split, same generate-pick-explain
mechanics as the existing pipeline (evaluate_emcqa.py / score_explanations.py work on
this file unchanged) — only the content of the question and the four choices differs:
each choice is a full analysis-style ANSWER (not a short fact), one is correct/
well-reasoned and three are specifically, deliberately flawed.

Scoped to the 3 categories with a well-defined correct/incorrect structure:
  - premise_correction:    the embedded claim is either right or wrong
  - methodology_critique:  a specific real design flaw exists or doesn't
  - result_interpretation: one correct read of the result vs. plausible-but-wrong ones
(hypothesis_generation / research_consult / gene_target_prioritization deliberately
excluded — "best answer" is too fuzzy there for a forced single-choice format to be honest.)

Usage:
    python generate_aligned_mcq.py \
        --corpus /uss/skavlak/version_3/corpus_v3_eval.jsonl \
        --out    /uss/skavlak/version_3/aligned_mcq_eval.jsonl \
        --n-per-category 15
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

CATEGORY_INSTRUCTIONS = {
    "premise_correction": (
        "Given a researcher's question that embeds an assumption or claim, do not accept "
        "the claim at face value. Use your knowledge of M. tuberculosis to assess whether "
        "it is correct, and steer the researcher in the right direction."
    ),
    "methodology_critique": (
        "The user describes a planned or completed experimental approach involving "
        "M. tuberculosis. Critically evaluate the methodology: identify missing controls, "
        "confounds, or design flaws, and suggest concrete improvements."
    ),
    "result_interpretation": (
        "Given a researcher's description of an experiment and their observations, explain "
        "the likely biological significance of the results based on your knowledge of "
        "M. tuberculosis."
    ),
}

SYSTEM_PROMPT = """\
You are an expert in tuberculosis (TB) research with deep knowledge of microbiology, \
molecular biology, genomics, drug resistance, host-pathogen interactions, and TB \
therapeutics. You are building an evaluation item to test whether a fine-tuned TB \
research-assistant LLM can genuinely do open-ended analytical reasoning — not just \
recall facts — in the style of: "{category}"."""

ITEM_PROMPT = """\
Category instruction this question must match: "{category_instruction}"

Using the passage below as grounding, write ONE realistic researcher scenario/question in \
this category's style (a researcher describing their work/data/plan and asking for help), \
plus FOUR full candidate ANSWERS to it — each a genuine paragraph-length analysis response, \
not a short fact or phrase. Exactly ONE answer must be correct/well-reasoned. The other \
THREE must each be wrong in one SPECIFIC, clearly identifiable way:
  - one that states something factually incorrect about the biology
  - one that misses the actual key issue (engages with the scenario but focuses on the \
wrong thing, or reaches a conclusion the evidence doesn't support)
  - one that is generic/evasive — sounds reasonable but doesn't actually engage with the \
specifics of this scenario, or hedges without committing to an actual answer

The question should require real analysis, not something answerable by pattern-matching \
words from the passage. The four candidate answers should all be plausible-sounding and \
similar in length/tone — a reader shouldn't be able to spot the correct one just because \
it's longer or more confident.

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
  "explanation": "The correct answer is A because ... Choice B is wrong because [factually incorrect: ...]. Choice C is wrong because [misses the key issue: ...]. Choice D is wrong because [generic/evasive: ...].",
  "topic": "drug_resistance"
}}"""

VALID_TOPICS = {
    "drug_resistance", "host_pathogen", "virulence", "genomics", "transcriptomics",
    "metabolomics", "protein_structure", "drug_targets", "signal_transduction",
    "epigenomics", "biofilm", "latency_dormancy", "vaccine", "diagnosis", "clinical",
}


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
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_item(item: dict) -> bool:
    required = {"question", "choices", "correct", "explanation", "topic"}
    if not required.issubset(item.keys()):
        return False
    if set(item["choices"].keys()) != {"A", "B", "C", "D"}:
        return False
    if item["correct"] not in {"A", "B", "C", "D"}:
        return False
    if len(item["question"].strip()) < 20:
        return False
    if item["topic"] not in VALID_TOPICS:
        return False
    for v in item["choices"].values():
        if len(v.strip()) < 40:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/uss/skavlak/version_3/corpus_v3_eval.jsonl")
    parser.add_argument("--out", default="/uss/skavlak/version_3/aligned_mcq_eval.jsonl")
    parser.add_argument("--n-per-category", type=int, default=15)
    parser.add_argument("--categories", default=",".join(CATEGORY_INSTRUCTIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    categories = args.categories.split(",")
    for c in categories:
        if c not in CATEGORY_INSTRUCTIONS:
            raise SystemExit(f"Unknown category: {c}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_chunk_ids = set()
    counts = {c: 0 for c in categories}
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_chunk_ids.add(obj.get("source_chunk_id", ""))
                lv = obj.get("cognitive_level")
                if lv in counts:
                    counts[lv] += 1
        print(f"Resuming: {dict(counts)}")

    n_total_target = args.n_per_category * len(categories)
    chunks = load_chunks(args.corpus, n_total_target * 3, args.seed)  # oversample for failures
    print(f"Loaded {len(chunks)} candidate chunks")

    generated = sum(counts.values())
    failed = 0
    chunk_i = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for category in categories:
            cat_instruction = CATEGORY_INSTRUCTIONS[category]
            while counts[category] < args.n_per_category and chunk_i < len(chunks):
                chunk = chunks[chunk_i]
                chunk_i += 1
                chunk_id = chunk.get("id", chunk.get("document_name", ""))
                if chunk_id in done_chunk_ids:
                    continue

                passage = chunk.get("text", "")[:2000]
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT.format(category=category)},
                            {"role": "user", "content": ITEM_PROMPT.format(
                                category_instruction=cat_instruction, passage=passage)},
                        ],
                        temperature=0.8,
                        max_tokens=1600,
                        response_format={"type": "json_object"},
                    )
                    raw = response.choices[0].message.content
                    item = parse_response(raw)

                    if item is None or not validate_item(item):
                        print(f"  [skip] invalid item for chunk {chunk_id} ({category})")
                        failed += 1
                        continue

                    record = {
                        "id": f"amcq_{category}_{counts[category]:03d}",
                        "source_chunk_id": chunk_id,
                        "document_name": chunk.get("document_name", ""),
                        "cognitive_level": category,  # repurposed field: holds SFT category
                        **item,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    counts[category] += 1
                    generated += 1
                    print(f"  [{generated}/{n_total_target}] {category} {counts[category]}/{args.n_per_category}")
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  [error] {e}")
                    failed += 1
                    time.sleep(2)

    print(f"\nDone. {generated} items saved to {out_path} ({failed} failed/skipped)")
    print(f"Final counts: {dict(counts)}")


if __name__ == "__main__":
    main()
