#!/usr/bin/env python3
"""
rag_pipeline_medimind.py — Retrieve-then-generate pipeline paired with the
MediMind-style index (build_index_medimind.py / merge_index_medimind.py):
FAISS/HNSW + jina-embeddings-v3 late-chunking retrieval, generation with our
own v5 SFT checkpoint (best standalone condition on the aligned-960 eval).

Must run inside /uss/skavlak/envs/jina_rag_env (jina-v3's custom remote code
needs transformers==4.57.3-compatible internals; the project's main venv was
briefly, accidentally mismatched with this during setup — see session notes).
That env also already has bitsandbytes/peft/accelerate, so retrieval and
generation both run in this one process without crossing environments.

Importable as a module (load_index / load_generator / query_rag), or run
directly via chat_rag_medimind.py for a CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import bm25s
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from build_index_medimind import LateChunkEmbedder, EMBEDDING_MODEL, OUT_DIR, INDEX_PATH, METADATA_PATH

RAG_TOP_K = 8
BM25_INDEX_DIR = OUT_DIR / "bm25_index"

# How many candidates each retriever contributes before fusion — wider than the
# final top_k so RRF has enough overlap/signal to work with.
FUSION_DEPTH = 50
RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al. 2009)

SYSTEM_PROMPT = """\
You are an expert tuberculosis (TB) research assistant. You have access to a curated corpus \
of TB research literature (context chunks below). Each chunk comes from a source document — \
cite it naturally using the document name when you use it. Prioritise answering from the \
context. The retrieved text may itself contain inline numeric or bracketed citation markers \
from the original paper's own reference list (e.g. "(39)", "[12]") — you were NOT given that \
paper's reference list, so these numbers are meaningless to you and to the reader. Never \
reproduce a bare numeric/bracketed citation marker from the retrieved text; if you need to \
refer to the claim it was attached to, cite the document it came from instead, or drop the \
marker entirely. If the context chunks are not relevant to the question, answer from your \
general TB/biomedical knowledge but clearly note: "This is from general knowledge, not the \
specific literature." Never fabricate citations.

Format your answer for readability using Markdown: keep paragraphs short, use a bullet list \
("- item") when enumerating multiple options, steps, or causes, use **bold** for key terms you \
are defining or emphasising, and use ## or ### headers only for longer answers with multiple \
distinct parts. Do not force structure (headers/bullets) onto a short, single-point answer that \
doesn't need it."""

_embedder: Optional[LateChunkEmbedder] = None
_index = None
_metadata: Optional[list] = None
_bm25 = None
_model = None
_tokenizer = None


def load_index():
    global _index, _metadata
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{INDEX_PATH} not found — run build_index_medimind.py / merge_index_medimind.py first."
            )
        _index = faiss.read_index(str(INDEX_PATH))
        _index.hnsw.efSearch = 64
        _metadata = [json.loads(l) for l in METADATA_PATH.open("r", encoding="utf-8")]
        assert _index.ntotal == len(_metadata), (
            f"index/metadata count mismatch: {_index.ntotal} vectors vs {len(_metadata)} metadata rows"
        )
        print(f"MediMind FAISS index loaded — {_index.ntotal} chunks indexed")
    return _index, _metadata


def load_embedder() -> LateChunkEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LateChunkEmbedder(EMBEDDING_MODEL)
    return _embedder


def load_bm25():
    global _bm25
    if _bm25 is None:
        if not BM25_INDEX_DIR.exists():
            raise FileNotFoundError(
                f"{BM25_INDEX_DIR} not found — run build_bm25_medimind.py first."
            )
        _bm25 = bm25s.BM25.load(str(BM25_INDEX_DIR), load_corpus=False)
        print(f"BM25 index loaded from {BM25_INDEX_DIR}")
    return _bm25


def load_generator(model_path: str, tokenizer_path: Optional[str] = None, adapter_path: Optional[str] = None):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    tok_path = tokenizer_path or model_path
    _tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left", truncation_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if adapter_path:
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto",
            attn_implementation="sdpa",
        )
        _model = PeftModel.from_pretrained(base, adapter_path)
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
    _model.eval()
    return _model, _tokenizer


def _dense_search(question: str, depth: int) -> list[int]:
    """Returns chunk indices ranked by dense (jina-v3) similarity, best first."""
    index, _ = load_index()
    embedder = load_embedder()
    q_vec = embedder.embed_query(question)
    _, I = index.search(np.expand_dims(q_vec, 0), depth)
    return [int(idx) for idx in I[0] if idx != -1]


def _bm25_search(question: str, depth: int) -> list[int]:
    """Returns chunk indices ranked by BM25 keyword score, best first."""
    bm25 = load_bm25()
    query_tokens = bm25s.tokenize(question, stopwords="english", show_progress=False)
    docs, _ = bm25.retrieve(query_tokens, k=depth, show_progress=False)
    return [int(idx) for idx in docs[0]]


def _reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Standard RRF: score(doc) = sum over lists of 1 / (k + rank), rank is 0-based."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def retrieve(question: str, top_k: int = RAG_TOP_K, mode: str = "hybrid") -> list:
    """Returns a list of metadata dicts (document_name, text, char_start/end), most relevant first.

    mode:
        "hybrid" (default) — dense (jina-v3) + BM25 keyword search, fused via Reciprocal Rank Fusion.
        "dense"  — jina-v3 embedding similarity only (the original behavior).
        "bm25"   — keyword search only.
    """
    _, metadata = load_index()

    if mode == "dense":
        idxs = _dense_search(question, top_k)
    elif mode == "bm25":
        idxs = _bm25_search(question, top_k)
    elif mode == "hybrid":
        dense_ranked = _dense_search(question, FUSION_DEPTH)
        bm25_ranked = _bm25_search(question, FUSION_DEPTH)
        idxs = _reciprocal_rank_fusion([dense_ranked, bm25_ranked])[:top_k]
    else:
        raise ValueError(f"Unknown retrieval mode: {mode!r} (expected dense/bm25/hybrid)")

    return [metadata[idx] for idx in idxs]


def build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    return "\n\n---\n\n".join(f"[Source: {r['document_name']}]\n{r['text']}" for r in retrieved)


def query_rag(question: str, model_path: str, tokenizer_path: Optional[str] = None,
              adapter_path: Optional[str] = None, top_k: int = RAG_TOP_K,
              max_new_tokens: int = 512, mode: str = "hybrid") -> tuple[str, list]:
    """Retrieve relevant TB corpus chunks (MediMind-style index) and generate
    a grounded answer with the given (default: v5) generation model.

    Returns (answer, retrieved_metadata).
    """
    model, tokenizer = load_generator(model_path, tokenizer_path, adapter_path)
    retrieved = retrieve(question, top_k=top_k, mode=mode)
    context = build_context(retrieved)

    format_reminder = (
        "(Format this answer in Markdown: short paragraphs, a bullet list for "
        "multiple options/steps/causes, **bold** for key terms, ## headers only "
        "if it's a long multi-part answer.)"
    )
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n{format_reminder}\n\nAnswer:"
        if context else
        f"Question: {question}\n\n{format_reminder}\n\nAnswer:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return answer, retrieved
