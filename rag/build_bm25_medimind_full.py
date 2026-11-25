#!/usr/bin/env python3
"""
build_bm25_medimind_full.py — "Upper bound" variant of build_bm25_medimind.py:
builds the BM25 keyword index over the full/upper-bound MediMind corpus
(291,387 chunks, including the 200 held-out eval papers), to pair with
rag_pipeline_medimind_full.py's dense (jina-v3/FAISS) retrieval for hybrid
mode. Row i of this index must stay aligned with row i of
tb_rag_medimind_faiss_full/metadata.jsonl, exactly as in the lower-bound
build_bm25_medimind.py.

Usage:
    /uss/skavlak/envs/jina_rag_env/bin/python3 build_bm25_medimind_full.py
"""

import json
from pathlib import Path

import bm25s

OUT_DIR = Path("/uss/skavlak/tb_rag_medimind_faiss_full")
METADATA_PATH = OUT_DIR / "metadata.jsonl"
BM25_INDEX_DIR = OUT_DIR / "bm25_index"


def main():
    print(f"Loading chunk texts from {METADATA_PATH}...")
    texts = []
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"{len(texts)} chunks loaded")

    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(texts, stopwords="english", show_progress=True)

    print("Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)

    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(BM25_INDEX_DIR))
    print(f"Saved BM25 index to {BM25_INDEX_DIR}")


if __name__ == "__main__":
    main()
