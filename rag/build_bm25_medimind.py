#!/usr/bin/env python3
"""
build_bm25_medimind.py — Build a BM25 keyword-search index over the same
289,172 chunks as the MediMind-style FAISS index (index.faiss / metadata.jsonl),
for hybrid (dense + keyword) retrieval in rag_pipeline_medimind.py.

Row i of metadata.jsonl corresponds to row i of the FAISS index AND to row i
of this BM25 index — all three are kept in the same order, so an index id
returned by either retriever maps directly to metadata[id].

Usage:
    /uss/skavlak/envs/jina_rag_env/bin/python3 build_bm25_medimind.py
"""

import json
from pathlib import Path

import bm25s

OUT_DIR = Path("/uss/skavlak/tb_rag_medimind_faiss")
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
