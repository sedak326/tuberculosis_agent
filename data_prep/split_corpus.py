#!/usr/bin/env python3
"""
split_corpus.py — Merge old and new corpora, hold out a fixed number of papers for eval.

Usage:
    python split_corpus.py \
        --old  /home/skavlak/finetuning/mtubercolosis/output/corpus.jsonl \
        --new  /uss/skavlak/tb_corpus_v3/corpus_new.jsonl \
        --out  /uss/skavlak/tb_corpus_v3
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old",  required=True, help="Existing proteomics corpus.jsonl")
    parser.add_argument("--new",  required=True, help="New broader corpus.jsonl")
    parser.add_argument("--out",  required=True, help="Output directory")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of papers to hold out for MCQ eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all chunks, grouped by document
    doc_chunks = defaultdict(list)
    total = 0
    for path in [args.old, args.new]:
        print(f"Loading {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                doc = chunk.get("document_name", chunk.get("document", "unknown"))
                doc_chunks[doc].append(chunk)
                total += 1

    print(f"  {total} chunks across {len(doc_chunks)} documents")

    # Hold out exactly n_eval papers; rest goes to training
    docs = sorted(doc_chunks.keys())
    random.seed(args.seed)
    random.shuffle(docs)

    n_eval = min(args.n_eval, len(docs))
    eval_docs  = set(docs[:n_eval])
    train_docs = set(docs[n_eval:])

    print(f"  Train docs: {len(train_docs)}  Eval docs: {len(eval_docs)} (held out for MCQ)")

    train_path = out_dir / "corpus_v3_train.jsonl"
    eval_path  = out_dir / "corpus_v3_eval.jsonl"

    train_chunks = eval_chunks = 0
    with open(train_path, "w", encoding="utf-8") as tf, \
         open(eval_path,  "w", encoding="utf-8") as ef:
        for doc, chunks in doc_chunks.items():
            if doc in eval_docs:
                for c in chunks:
                    ef.write(json.dumps(c, ensure_ascii=False) + "\n")
                eval_chunks += len(chunks)
            else:
                for c in chunks:
                    tf.write(json.dumps(c, ensure_ascii=False) + "\n")
                train_chunks += len(chunks)

    print(f"  Train chunks: {train_chunks} → {train_path}")
    print(f"  Eval chunks : {eval_chunks}  → {eval_path}")


if __name__ == "__main__":
    main()
