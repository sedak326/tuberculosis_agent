#!/usr/bin/env python3
"""
train_cpt.py — Continued pre-training of a LLaMA base model on TB literature.

Causal language modelling loss over all tokens (no response masking).
Uses sequence packing so no tokens are wasted on padding.

Full-parameter CPT (via SLURM — see run_cpt.slurm):
    torchrun --nproc_per_node=2 train_cpt.py

QLoRA CPT (DDP — each GPU holds the full quantized model):
    torchrun --nproc_per_node=4 train_cpt.py --bits 4 --lora-rank 64

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

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

CORPUS_PATH = "/home/skavlak/finetuning/mtubercolosis/output/corpus.jsonl"
MODEL_ID    = "meta-llama/Llama-3.1-8B"   # base model, NOT instruct
OUTPUT_DIR  = "/uss/skavlak/tb_cpt_checkpoints"

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
    parser.add_argument("--optim",       type=str,   default="adamw_torch",
                        help="Optimizer (e.g. adamw_bnb_8bit for memory-constrained 70B runs)")
    parser.add_argument("--adam-beta1",  type=float, default=0.9,
                        help="Adam beta1 (first moment decay)")
    parser.add_argument("--adam-beta2",  type=float, default=0.999,
                        help="Adam beta2 (second moment decay). Scale with batch size to hold the "
                             "second-moment half-life constant in tokens (Marek et al. 2025): "
                             "beta2_new = beta2_ref ** (batch_new / batch_ref)")
    parser.add_argument("--lr-scheduler-type", default="cosine",
                        help="LR schedule (e.g. 'constant' to disable warmup/decay entirely)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Ignored when --lr-scheduler-type is 'constant'")
    parser.add_argument("--bits",        type=int,   default=None, choices=[4, 8],
                        help="Load model in N-bit quantization for QLoRA (requires --lora-rank).")
    parser.add_argument("--lora-rank",   type=int,   default=None,
                        help="Enable LoRA fine-tuning with this rank. When set, FSDP is disabled "
                             "and each GPU holds the full (optionally quantized) model.")
    args = parser.parse_args()

    train_ds, val_ds = load_corpus(args.corpus, sample=args.sample)
    print(f"Train chunks: {len(train_ds)}  Val chunks: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    use_lora = args.lora_rank is not None
    # LoRA (optionally quantized) always uses DDP, each GPU holding the full
    # model — mirrors train_sft.py's QLoRA path exactly. FSDP is only for
    # full-parameter training.
    use_fsdp = distributed and not use_lora

    # Load model explicitly only when quantization is requested; otherwise pass
    # the model ID string and let SFTTrainer handle loading (preserves FSDP path).
    if args.bits is not None:
        if not use_lora:
            raise ValueError("--bits requires --lora-rank")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(args.bits == 4),
            load_in_8bit=(args.bits == 8),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            dtype=torch.bfloat16,
        )
        from peft import prepare_model_for_kbit_training
        # Don't enable gradient checkpointing here — SFTConfig handles it below
        # with use_reentrant=False, which is required for DDP + frozen base params.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = args.model

    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        bf16=True,
        # With FSDP, use activation_checkpointing inside fsdp_config instead.
        # For single-GPU or LoRA/DDP runs, gradient_checkpointing here is fine
        # (LoRA needs use_reentrant=False for frozen base params).
        gradient_checkpointing=not use_fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_lora else {},
        ddp_find_unused_parameters=False if use_lora else None,
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
        fsdp="full_shard auto_wrap" if use_fsdp else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "backward_prefetch": "backward_pre",
            "use_orig_params": True,
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "activation_checkpointing": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        } if use_fsdp else {},
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume = args.resume_from if args.resume_from != "latest" else True
    trainer.train(resume_from_checkpoint=resume)

    final_dir = str(Path(args.output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"CPT model saved to {final_dir}")


if __name__ == "__main__":
    main()
