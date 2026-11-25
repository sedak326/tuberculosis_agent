#!/usr/bin/env python3
"""
merge_index_medimind.py — Concatenate the per-shard vectors/metadata from
build_index_medimind.py (run with --num-shards > 1) into one combined
FAISS/HNSW index + metadata.jsonl.

Usage:
    python merge_index_medimind.py --num-shards 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

OUT_DIR       = Path("/uss/skavlak/tb_rag_medimind_faiss")
INDEX_PATH    = OUT_DIR / "index.faiss"
METADATA_PATH = OUT_DIR / "metadata.jsonl"
EMBED_DIM     = 1024


def merge(num_shards: int) -> None:
    all_vecs = []
    total = 0
    with METADATA_PATH.open("w", encoding="utf-8") as out_meta:
        for shard_id in range(num_shards):
            suffix = f"_shard{shard_id}of{num_shards}"
            vecs_path = OUT_DIR / f"vectors{suffix}.npy"
            meta_path = OUT_DIR / f"metadata{suffix}.jsonl"
            vecs = np.load(vecs_path)
            all_vecs.append(vecs)
            n_meta = 0
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    out_meta.write(line)
                    n_meta += 1
            print(f"shard {shard_id}: {vecs.shape[0]} vectors, {n_meta} metadata rows")
            assert vecs.shape[0] == n_meta, f"shard {shard_id} vectors/metadata count mismatch"
            total += n_meta

    combined = np.concatenate(all_vecs, axis=0)
    print(f"\nTotal: {total} chunks, vectors shape {combined.shape}")

    index = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(combined)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved merged FAISS index to {INDEX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    merge(args.num_shards)
