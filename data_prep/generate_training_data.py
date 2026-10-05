#!/usr/bin/env python3
"""
generate_training_data.py — Synthetic instruction-response pair generator

Reads corpus.jsonl, sends each body text chunk to a local vLLM server along
with the seed task examples, and writes instruction-response pairs in LLaMA
chat format to output/training_data.jsonl.

Usage:
    # Spot-check: generate from 10 chunks only
    python generate_training_data.py --sample 10

    # Full run
    python generate_training_data.py

    # Resume after interruption (already-processed chunks are skipped)
    python generate_training_data.py

    # Parallel shard (e.g. shard 2 of 4, pointing at port 8001)
    python generate_training_data.py --num-shards 4 --shard-id 2 --base-url http://localhost:8001/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SEED_PATH   = Path(__file__).parent / "seeds.json"
CORPUS_PATH = Path("/uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl")
OUTPUT_PATH = Path("/uss/skavlak/tb_corpus_v3/training_data_v3.jsonl")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EXAMPLES = 2
MIN_CHUNK_CHARS  = 300
MAX_CHUNK_CHARS  = 3000
RETRY_SLEEP      = 5

SYSTEM_PROMPT = """\
You are generating instruction-response pairs for supervised fine-tuning of a tuberculosis \
research assistant LLM. The final model will assist TB researchers in their day-to-day work — \
answering factual questions about genes, drugs, and mechanisms; interpreting experimental \
results; generating hypotheses; advising on target prioritization; explaining dense literature; \
and helping researchers connect their expertise to TB biology. Study the seed examples \
carefully — they define the exact format, tone, and task type you must follow.

The examples should feel like a TB domain expert advising a fellow researcher, not a textbook \
or encyclopedia entry.

All {n} examples you generate must belong to the SAME task category: "{category}". The seed \
examples below are all drawn from this category — match their instruction framing, tone, and \
structure closely. Do not drift into a different category (e.g. do not write a closed-book-QA \
style example when the category is methodology_critique).

Format each example exactly like this:

[number].
Instruction: [Generic category-level task framing, matching the "{category}" seed examples' \
instruction style. Do not write question-specific instructions like "Explain the role of X".]
Input: [A naturalistic, first-person researcher question or scenario inspired by the passage, \
matching what this category's seeds look like. Write as a researcher speaking to a colleague — \
casual and direct, e.g. "Hey, I'm studying X and I want to understand Y...". Do not paste or \
paraphrase the source passage verbatim. It should be something a real TB researcher would \
plausibly ask.]
Output: [A response matching this category's style and purpose (see seed examples). Be \
specific: name genes, mechanisms, and relevant findings where applicable. Write 150-300 words.]

Rules:
- Follow the seed instruction templates for this category closely for the Instruction field.
- The Input must sound like a real researcher talking — casual, first-person, specific.
- Ground factual claims in what the passage tells you, but write the Output as if answering \
from memory and expertise.
- Return exactly {n} examples, all in the "{category}" category, using the numbered format above.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_tasks() -> list[dict]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def format_seeds(seeds: list[dict], category: str, max_n: int = 8) -> str:
    """Return only seeds from the given category, so few-shot guidance stays on-category
    regardless of how unevenly seeds are distributed across categories."""
    matching = [s for s in seeds if s.get("category") == category]
    sample = random.sample(matching, min(max_n, len(matching)))
    lines = []
    for s in sample:
        lines.append(f"### {s.get('category', '')}")
        lines.append(f"Instruction: {s.get('instruction', '')}")
        inp = s.get("input", "")
        if inp:
            lines.append(f"Input: {inp}")
        lines.append(f"Output: {s.get('output', '')}")
        lines.append("")
    return "\n".join(lines)


def categories_from_seeds(seeds: list[dict]) -> list[str]:
    return sorted({s.get("category", "") for s in seeds if s.get("category")})


def load_chunks(corpus_path: Path) -> list[dict]:
    chunks = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("content_type") != "text":
                continue
            if obj.get("section_title") == "Abstract":
                continue
            raw  = obj.get("text", "")
            body = raw[raw.index("\n\n") + 2:] if "\n\n" in raw else raw
            if len(body) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id":       obj.get("id", ""),
                "document": obj.get("document_name", ""),
                "section":  obj.get("section_title", ""),
                "body":     body,
            })
    return chunks


def already_processed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cid = obj.get("source_chunk_id")
                if cid:
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    return ids


def parse_gpt_response(raw: str) -> list[dict]:
    examples = []
    blocks = re.split(r"\n(?=\d+[\.\)])", raw.strip())
    for block in blocks:
        inst = re.search(
            r"\*{0,2}Instruction:?\*{0,2}\s*(.+?)(?=\n\*{0,2}Input:|\Z)",
            block, re.DOTALL | re.IGNORECASE,
        )
        inp = re.search(
            r"\*{0,2}Input:?\*{0,2}\s*(.+?)(?=\n\*{0,2}Output:|\Z)",
            block, re.DOTALL | re.IGNORECASE,
        )
        out = re.search(
            r"\*{0,2}Output:?\*{0,2}\s*(.+?)(?=\n\d+[\.\)]|\Z)",
            block, re.DOTALL | re.IGNORECASE,
        )
        if inst and out:
            examples.append({
                "instruction": inst.group(1).strip(),
                "input":       inp.group(1).strip() if inp else "",
                "output":      out.group(1).strip(),
            })
    return examples


def to_chat_record(example: dict, source_chunk_id: str, category: str) -> dict:
    user_content = example["instruction"]
    if example["input"]:
        user_content += "\n\n" + example["input"]
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ],
        "source_chunk_id": source_chunk_id,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def call_model(
    client:     AsyncOpenAI,
    chunk:      dict,
    seed_tasks: list[dict],
    category:   str,
    n:          int,
    model:      str,
) -> str:
    user_msg = (
        f"Here are example task types to follow (all category \"{category}\"):\n\n"
        f"{format_seeds(seed_tasks, category)}\n\n"
        f"---\n\n"
        f"Now generate {n} new \"{category}\" training examples grounded in the following "
        f"TB research text.\n\n"
        f"Paper: {chunk['document']}\n"
        f"Section: {chunk['section']}\n\n"
        f"Text:\n{chunk['body'][:MAX_CHUNK_CHARS]}"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n, category=category)},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TB training data from corpus.jsonl")
    parser.add_argument("--sample",            type=int,   default=None)
    parser.add_argument("--examples-per-chunk",type=int,   default=DEFAULT_EXAMPLES)
    parser.add_argument("--model",                         default=DEFAULT_MODEL)
    parser.add_argument("--corpus",                        default=str(CORPUS_PATH))
    parser.add_argument("--output",                        default=str(OUTPUT_PATH))
    parser.add_argument("--base-url",                      default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency",       type=int,   default=4,
                        help="Number of concurrent requests to vLLM (default: 4)")
    parser.add_argument("--num-shards",        type=int,   default=1,
                        help="Total number of parallel shards (default: 1 = no sharding)")
    parser.add_argument("--shard-id",          type=int,   default=0,
                        help="Which shard this process handles (0-indexed)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    # When sharding, write to a shard-specific file to avoid conflicts
    if args.num_shards > 1:
        output_path = output_path.parent / f"training_data_shard_{args.shard_id}.jsonl"

    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client     = AsyncOpenAI(base_url=args.base_url, api_key="ollama")
    seed_tasks = load_seed_tasks()
    all_chunks = load_chunks(corpus_path)

    # Shard by original corpus index (stable across restarts)
    if args.num_shards > 1:
        all_chunks = [c for i, c in enumerate(all_chunks) if i % args.num_shards == args.shard_id]

    done_ids = already_processed_ids(output_path)
    pending  = [c for c in all_chunks if c["id"] not in done_ids]
    if args.sample:
        pending = pending[:args.sample]

    # Round-robin assign each chunk a target category so the generated output is evenly
    # balanced across categories, regardless of how unevenly the seed pool itself is sized.
    # Shuffle first (deterministic seed) so category assignment isn't correlated with
    # corpus file order (e.g. all "drug resistance" chunks landing in the same category).
    categories = categories_from_seeds(seed_tasks)
    rng = random.Random(42)
    shuffled = pending[:]
    rng.shuffle(shuffled)
    chunk_category = {c["id"]: categories[i % len(categories)] for i, c in enumerate(shuffled)}

    print(f"Shard                   : {args.shard_id}/{args.num_shards}")
    print(f"Corpus chunks in shard  : {len(all_chunks)}")
    print(f"Already processed       : {len(done_ids)}")
    print(f"To process              : {len(pending)}")
    print(f"Categories ({len(categories)})       : {categories}")
    print(f"Examples per chunk      : {args.examples_per_chunk}")
    print(f"Estimated total examples: {len(pending) * args.examples_per_chunk}")
    print(f"Concurrency             : {args.concurrency}")
    print(f"Model                   : {args.model}")
    print(f"Base URL                : {args.base_url}")
    print(f"Output                  : {output_path}")
    print()

    written   = 0
    failed    = 0
    completed = 0
    lock      = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    with output_path.open("a", encoding="utf-8") as out_f:

        async def process_one(chunk: dict) -> None:
            nonlocal written, failed, completed
            category = chunk_category[chunk["id"]]
            async with semaphore:
                for attempt in range(2):
                    try:
                        raw      = await call_model(client, chunk, seed_tasks, category, args.examples_per_chunk, args.model)
                        examples = parse_gpt_response(raw)
                        async with lock:
                            completed += 1
                            if not examples:
                                print(f"  WARNING: no examples parsed from chunk {chunk['id']}")
                                failed += 1
                                return
                            for ex in examples:
                                out_f.write(json.dumps(to_chat_record(ex, chunk["id"], category), ensure_ascii=False) + "\n")
                            written += len(examples)
                            out_f.flush()
                            if completed % 25 == 0 or completed == len(pending):
                                print(f"  [{completed}/{len(pending)}] {written} examples written, {failed} failed")
                        return
                    except Exception as exc:
                        if attempt == 0:
                            await asyncio.sleep(RETRY_SLEEP)
                        else:
                            async with lock:
                                completed += 1
                                failed    += 1
                                print(f"  FAILED chunk {chunk['id']}: {exc}")

        try:
            await asyncio.gather(*[process_one(chunk) for chunk in pending])
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved — rerun to resume.")

    print(f"\nFinished. {written} examples written to {output_path}")
    if failed:
        print(f"  {failed} chunks failed — rerun to retry them")


if __name__ == "__main__":
    asyncio.run(main())
