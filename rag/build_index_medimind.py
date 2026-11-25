#!/usr/bin/env python3
"""
build_index_medimind.py — Build a FAISS/HNSW vector store using late chunking,
replicating MediMind's RAG architecture (Hashemi et al., IoT 2025) as closely
as practical on our own TB corpus:

  - Embedding model: jinaai/jina-embeddings-v3 (1024-dim, multilingual),
    using its "retrieval.passage" LoRA adapter for documents and
    "retrieval.query" for queries at search time.
  - Chunking: recursive-style splitting into 1000-token chunks with 200-token
    overlap (their Table/Section IV.D.2.b spec).
  - Late chunking (Günther et al. 2024, their reference [20]): chunk vectors
    are produced by mean-pooling token-level hidden states from a long-context
    encode of the *whole document*, not by embedding each chunk's text in
    isolation — preserves cross-chunk context.
  - Vector store: FAISS with HNSW indexing (their Section IV.D.2.c), not Chroma.

Deviation from the paper, by necessity: most of our source documents exceed
jina-v3's ~8192-token context window (confirmed: ~40% of a 60-doc sample),
which the paper's single-pass late-chunking description doesn't address. We
window long documents (8000 tokens/window, 200-token window overlap) and
assign each chunk to whichever window fully contains it, picking the most
centered option when a chunk falls in more than one window's overlap zone.
This keeps every chunk's late-chunking pool entirely within one window's
hidden states while still covering the full document.

Must run inside the isolated /uss/skavlak/envs/jina_rag_env conda env — the
jina-v3 custom remote code is incompatible with the transformers version used
by every other script in this project (see session notes).

Usage:
    conda activate /uss/skavlak/envs/jina_rag_env
    python build_index_medimind.py [--limit N] [--batch-size B]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

CORPUS_PATH   = Path("/uss/skavlak/version_5/corpus_v5_train.jsonl")
OUT_DIR       = Path("/uss/skavlak/tb_rag_medimind_faiss")
INDEX_PATH    = OUT_DIR / "index.faiss"
METADATA_PATH = OUT_DIR / "metadata.jsonl"
FLUSH_EVERY_DOCS = 100  # periodic checkpoint cadence for resumability

EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
EMBED_DIM       = 1024

CHUNK_SIZE_TOKENS    = 1000
CHUNK_OVERLAP_TOKENS = 200
CHUNK_STRIDE_TOKENS  = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS  # 800

WINDOW_SIZE_TOKENS   = 8000   # headroom under jina-v3's 8194 max_position_embeddings
WINDOW_OVERLAP_TOKENS = 200
WINDOW_STRIDE_TOKENS  = WINDOW_SIZE_TOKENS - WINDOW_OVERLAP_TOKENS  # 7800

HEADER_RE = re.compile(r"^\[Document:.*?\]\n\n")


# ---------------------------------------------------------------------------
# Document reconstruction (our existing corpus is already pre-chunked by a
# different, chapter/section-aware splitter; late chunking needs the whole
# document, so we rebuild it from the ordered original chunks).
# ---------------------------------------------------------------------------

def load_documents(jsonl_path: Path) -> Dict[str, str]:
    by_doc: Dict[str, List[Tuple[int, str]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            name = item["document_name"]
            m = re.search(r"chunk_(\d+)", item["id"])
            order = int(m.group(1)) if m else 0
            text = HEADER_RE.sub("", item.get("text") or "").strip()
            if text:
                by_doc.setdefault(name, []).append((order, text))

    docs: Dict[str, str] = {}
    for name, parts in by_doc.items():
        parts.sort(key=lambda p: p[0])
        docs[name] = " ".join(t for _, t in parts)
    return docs


# ---------------------------------------------------------------------------
# Chunk / window grid
# ---------------------------------------------------------------------------

def make_spans(total_len: int, size: int, stride: int) -> List[Tuple[int, int]]:
    """Sliding [start, end) spans covering [0, total_len), last span flush
    with the end rather than running past it."""
    if total_len <= size:
        return [(0, total_len)]
    spans = []
    start = 0
    while True:
        end = min(start + size, total_len)
        spans.append((start, end))
        if end == total_len:
            break
        start += stride
    return spans


def assign_chunk_to_window(chunk: Tuple[int, int], windows: List[Tuple[int, int]]) -> int:
    """Index of the window that fully contains `chunk`, preferring the one
    whose center is closest to the chunk's center. Falls back to the window
    with the most overlap if none fully contains it (rare edge case)."""
    c_start, c_end = chunk
    c_center = (c_start + c_end) / 2
    best_idx, best_dist = None, None
    for i, (w_start, w_end) in enumerate(windows):
        if w_start <= c_start and c_end <= w_end:
            dist = abs(((w_start + w_end) / 2) - c_center)
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = i, dist
    if best_idx is not None:
        return best_idx
    # Fallback: most-overlapping window (shouldn't normally trigger)
    best_idx, best_overlap = 0, -1
    for i, (w_start, w_end) in enumerate(windows):
        overlap = max(0, min(c_end, w_end) - max(c_start, w_start))
        if overlap > best_overlap:
            best_idx, best_overlap = i, overlap
    return best_idx


# ---------------------------------------------------------------------------
# Embedding (late chunking)
# ---------------------------------------------------------------------------

class LateChunkEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device
        self.passage_task_id = self.model._adaptation_map["retrieval.passage"]
        self.query_task_id = self.model._adaptation_map["retrieval.query"]

    @torch.no_grad()
    def embed_windows(self, token_id_lists: List[List[int]]) -> torch.Tensor:
        """Batched forward pass over variable-length windows. Returns
        (batch, seqlen, hidden) last_hidden_state, right-padded, plus the
        attention mask used (caller already has lengths to slice with)."""
        max_len = max(len(t) for t in token_id_lists)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(token_id_lists), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(token_id_lists), max_len), dtype=torch.long)
        for i, ids in enumerate(token_id_lists):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, : len(ids)] = 1
        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)
        adapter_mask = torch.full(
            (len(token_id_lists),), self.passage_task_id, dtype=torch.int32, device=self.device
        )
        out = self.model.roberta(input_ids=input_ids, attention_mask=attn_mask, adapter_mask=adapter_mask)
        return out.last_hidden_state.float().cpu(), attn_mask.cpu()

    def pool_span(self, hidden: torch.Tensor, start: int, end: int) -> np.ndarray:
        span = hidden[start:end]  # (span_len, hidden)
        vec = span.mean(dim=0).numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    @torch.no_grad()
    def embed_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], task="retrieval.query")
        vec = np.asarray(emb[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(limit: int | None, batch_size: int, shard_id: int, num_shards: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{shard_id}of{num_shards}" if num_shards > 1 else ""
    vectors_path = OUT_DIR / f"vectors{suffix}.npy"
    metadata_path = OUT_DIR / f"metadata{suffix}.jsonl"
    done_docs_path = OUT_DIR / f"done_docs{suffix}.txt"

    print(f"Loading + reconstructing documents from {CORPUS_PATH}...")
    docs = load_documents(CORPUS_PATH)
    names = sorted(docs.keys())
    if num_shards > 1:
        names = [n for i, n in enumerate(names) if i % num_shards == shard_id]
    if limit:
        names = names[:limit]
    print(f"  {len(names)} documents assigned to shard {shard_id}/{num_shards}")

    # Resume: skip documents already recorded as done in a prior (killed/timed-out) run.
    already_done = set()
    if done_docs_path.exists():
        already_done = set(done_docs_path.read_text().splitlines())
        print(f"  Resuming: {len(already_done)} documents already done, skipping them")
    names = [n for n in names if n not in already_done]

    existing_vectors = []
    if vectors_path.exists():
        existing_vectors.append(np.load(vectors_path))
        print(f"  Loaded {existing_vectors[0].shape[0]} existing vectors from {vectors_path}")

    print(f"Loading {EMBEDDING_MODEL}...")
    embedder = LateChunkEmbedder(EMBEDDING_MODEL)
    tokenizer = embedder.tokenizer

    meta_f = metadata_path.open("a", encoding="utf-8")
    done_f = done_docs_path.open("a", encoding="utf-8")

    new_vectors: List[np.ndarray] = []
    docs_since_checkpoint: List[str] = []

    n_chunks_written = 0
    n_fallback = 0
    start_time = time.time()

    def checkpoint():
        """Persist vectors/metadata/done-list so a killed/resumed job stays consistent."""
        if new_vectors:
            existing_vectors.append(np.stack(new_vectors))
            combined = np.concatenate(existing_vectors, axis=0)
            np.save(vectors_path, combined)
            existing_vectors.clear()
            existing_vectors.append(combined)
            new_vectors.clear()
        meta_f.flush()
        for d in docs_since_checkpoint:
            done_f.write(d + "\n")
        done_f.flush()
        docs_since_checkpoint.clear()

    def embed_document(name: str) -> None:
        """Compute chunk/window spans for one document, embed all its windows
        in a single batched forward pass, and append results to new_vectors."""
        nonlocal n_chunks_written, n_fallback
        text = docs[name]
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        L = len(token_ids)
        if L == 0:
            return

        chunk_spans = make_spans(L, CHUNK_SIZE_TOKENS, CHUNK_STRIDE_TOKENS)
        window_spans = make_spans(L, WINDOW_SIZE_TOKENS, WINDOW_STRIDE_TOKENS)

        windows_chunks: List[List] = [[] for _ in window_spans]
        for c_idx, (c_start, c_end) in enumerate(chunk_spans):
            w_idx = assign_chunk_to_window((c_start, c_end), window_spans)
            w_start, w_end = window_spans[w_idx]
            if not (w_start <= c_start and c_end <= w_end):
                n_fallback += 1
                c_end = min(c_end, w_end)
                c_start = max(c_start, w_start)
            char_start = offsets[c_start][0]
            char_end = offsets[c_end - 1][1]
            chunk_text = text[char_start:char_end]
            windows_chunks[w_idx].append(
                (c_start - w_start, c_end - w_start, char_start, char_end, c_idx, chunk_text)
            )

        used_window_idxs = [i for i, wc in enumerate(windows_chunks) if wc]
        if not used_window_idxs:
            return
        token_lists = [token_ids[window_spans[i][0]:window_spans[i][1]] for i in used_window_idxs]

        # Sub-batch within a document in case it has more windows than fit in memory at once.
        for start in range(0, len(token_lists), max(1, batch_size)):
            sub_idxs = used_window_idxs[start:start + max(1, batch_size)]
            sub_tokens = token_lists[start:start + max(1, batch_size)]
            hidden_batch, _ = embedder.embed_windows(sub_tokens)
            for b_idx, w_idx in enumerate(sub_idxs):
                hidden = hidden_batch[b_idx]
                for local_start, local_end, char_start, char_end, chunk_idx, chunk_text in windows_chunks[w_idx]:
                    vec = embedder.pool_span(hidden, local_start, local_end)
                    new_vectors.append(vec)
                    meta_f.write(json.dumps({
                        "id": f"{name}_mmchunk_{chunk_idx:04d}",
                        "document_name": name,
                        "char_start": char_start,
                        "char_end": char_end,
                        "text": chunk_text,
                    }) + "\n")
                    n_chunks_written += 1

    for doc_i, name in enumerate(names):
        embed_document(name)
        docs_since_checkpoint.append(name)

        if len(docs_since_checkpoint) >= FLUSH_EVERY_DOCS or doc_i + 1 == len(names):
            checkpoint()
            elapsed = time.time() - start_time
            rate = (doc_i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(names) - doc_i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{doc_i + 1}/{len(names)}] docs, {n_chunks_written} chunks, "
                  f"{rate:.2f} docs/s, ETA {eta:.1f} min, fallback={n_fallback}", flush=True)

    meta_f.close()
    done_f.close()

    print(f"\nShard {shard_id}/{num_shards}: {n_chunks_written} chunks written "
          f"(fallback-clipped: {n_fallback})")
    print(f"Vectors saved to {vectors_path}, metadata to {metadata_path}")

    if num_shards == 1:
        print(f"Building FAISS index from {vectors_path}...")
        vecs = np.load(vectors_path)
        index = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(vecs)
        faiss.write_index(index, str(INDEX_PATH))
        print(f"Saved FAISS index ({vecs.shape[0]} vectors) to {INDEX_PATH}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents (smoke test)")
    parser.add_argument("--batch-size", type=int, default=4, help="Windows per forward-pass batch")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    build(args.limit, args.batch_size, args.shard_id, args.num_shards)
