#!/usr/bin/env python3
"""
train_cpt.py — Continued pre-training of a LLaMA base model on TB literature.

Causal language modelling loss over all tokens (no response masking).
Uses sequence packing so no tokens are wasted on padding.

Usage (via SLURM — see run_cpt.slurm):
    torchrun --nproc_per_node=2 train_cpt.py

Standalone test run (single GPU, no DeepSpeed):
    python train_cpt.py --epochs 1 --batch-size 1 --grad-accum 1 --sample 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

CORPUS_PATH = "/home/skavlak/finetuning/mtubercolosis/output/corpus.jsonl"
MODEL_ID    = "meta-llama/Llama-3.1-8B"   # base model, NOT instruct
OUTPUT_DIR  = "/home/skavlak/finetuning/mtubercolosis/output/tb_cpt_checkpoints"

# Strip the RAG metadata header added by extract_corpus.py:
# "[Document: ... | Chapter: ... | Section: ... | Page: ...]\n\n"
HEADER_RE = re.compile(r"^\[Document:.*?\]\n\n", re.DOTALL)


def strip_header(text: str) -> str:
    return HEADER_RE.sub("", text).strip()


def load_corpus(path: str, sample: int | None = None) -> tuple[Dataset, Dataset]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("content_type") != "text":
                continue
            body = strip_header(obj.get("text", ""))
            if len(body) >= 100:
                texts.append({"text": body})

    random.seed(42)
    random.shuffle(texts)

    if sample:
        texts = texts[:sample]

    # CPT doesn't need much validation data — 2% is enough to track perplexity
    n_val = max(1, int(len(texts) * 0.02))
    return Dataset.from_list(texts[n_val:]), Dataset.from_list(texts[:n_val])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",      default=CORPUS_PATH)
    parser.add_argument("--model",       default=MODEL_ID)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-seq-len", type=int,   default=2048)
    parser.add_argument("--sample",      type=int,   default=None,
                        help="Use only N chunks (for smoke-testing)")
    parser.add_argument("--resume-from", type=str,   default=None)
    args = parser.parse_args()

    train_ds, val_ds = load_corpus(args.corpus, sample=args.sample)
    print(f"Train chunks: {len(train_ds)}  Val chunks: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        # With FSDP, use activation_checkpointing inside fsdp_config instead.
        # For single-GPU runs, gradient_checkpointing here is fine.
        gradient_checkpointing=not distributed,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        dataset_text_field="text",
        packing=True,
        fsdp="full_shard auto_wrap" if distributed else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "backward_prefetch": "backward_pre",
            "use_orig_params": True,
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "activation_checkpointing": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        } if distributed else {},
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    resume = args.resume_from if args.resume_from != "latest" else True
    trainer.train(resume_from_checkpoint=resume)

    final_dir = str(Path(args.output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"CPT model saved to {final_dir}")


if __name__ == "__main__":
    main()
