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
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE        = Path(__file__).parent / "mtubercolosis"
SEED_PATH   = Path(__file__).parent / "seed_tasks.md"
CORPUS_PATH = HERE / "output" / "corpus.jsonl"
OUTPUT_PATH = HERE / "output" / "training_data.jsonl"

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
You are a biomedical expert helping build training data for a tuberculosis research assistant LLM.

You will be given a passage from a TB research paper. Generate {n} diverse instruction-response \
training examples grounded in that passage.

Rules:
- Use the seed examples provided as a guide for format and task diversity.
- The Input field must be a standalone question, gene name, drug name, or short clinical scenario. \
Do NOT copy or paraphrase the source passage into the Input.
- The Output must read as expert knowledge — not as a summary of the passage. Write as a TB \
researcher would answer from memory.
- Vary task types across your {n} examples: do not generate {n} copies of the same task type. \
Mix question answering, extraction, classification, explanation, comparison, summarisation, etc.
- Every factual claim in the Output must be supported by the source passage. Do not introduce \
information not present in the text.
- Return exactly {n} examples, numbered, each with Instruction / Input / Output fields.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_tasks() -> str:
    return SEED_PATH.read_text(encoding="utf-8")


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


def to_chat_record(example: dict, source_chunk_id: str) -> dict:
    user_content = example["instruction"]
    if example["input"]:
        user_content += "\n\n" + example["input"]
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ],
        "source_chunk_id": source_chunk_id,
    }


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def call_model(
    client:     AsyncOpenAI,
    chunk:      dict,
    seed_tasks: str,
    n:          int,
    model:      str,
) -> str:
    user_msg = (
        f"Here are example task types to follow:\n\n{seed_tasks}\n\n"
        f"---\n\n"
        f"Now generate {n} new training examples grounded in the following TB research text.\n\n"
        f"Paper: {chunk['document']}\n"
        f"Section: {chunk['section']}\n\n"
        f"Text:\n{chunk['body'][:MAX_CHUNK_CHARS]}"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=1000,
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

    print(f"Shard                   : {args.shard_id}/{args.num_shards}")
    print(f"Corpus chunks in shard  : {len(all_chunks)}")
    print(f"Already processed       : {len(done_ids)}")
    print(f"To process              : {len(pending)}")
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
            async with semaphore:
                for attempt in range(2):
                    try:
                        raw      = await call_model(client, chunk, seed_tasks, args.examples_per_chunk, args.model)
                        examples = parse_gpt_response(raw)
                        async with lock:
                            completed += 1
                            if not examples:
                                print(f"  WARNING: no examples parsed from chunk {chunk['id']}")
                                failed += 1
                                return
                            for ex in examples:
                                out_f.write(json.dumps(to_chat_record(ex, chunk["id"]), ensure_ascii=False) + "\n")
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
