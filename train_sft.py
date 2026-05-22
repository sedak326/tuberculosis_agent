#!/usr/bin/env python3
"""
train_sft.py — Full supervised fine-tuning of LLaMA 3.1 8B Instruct on TB training data.

Usage (via SLURM — see run_training.slurm):
    torchrun --nproc_per_node=2 train_sft.py --deepspeed deepspeed_zero2.json

Standalone test run (single GPU, no DeepSpeed):
    python train_sft.py --epochs 1 --batch-size 1 --grad-accum 1 --sample 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

DATA_PATH   = "/home/skavlak/finetuning/mtubercolosis/output/training_data_clean3.jsonl"
MODEL_ID    = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR  = "/home/skavlak/finetuning/mtubercolosis/output/tb_llm_checkpoints"


def load_splits(path: str, val_fraction: float = 0.05, sample: int | None = None):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records.append({"messages": obj["messages"]})

    random.seed(42)
    random.shuffle(records)

    if sample:
        records = records[:sample]

    n_val = max(1, int(len(records) * val_fraction))
    return Dataset.from_list(records[n_val:]), Dataset.from_list(records[:n_val])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=DATA_PATH)
    parser.add_argument("--model",       default=MODEL_ID)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-seq-len", type=int,   default=2048)
    parser.add_argument("--sample",      type=int,   default=None,
                        help="Use only N records (for smoke-testing)")
    parser.add_argument("--tokenizer",   default=None,
                        help="Tokenizer to use (defaults to --model). Override when model weights "
                             "come from a base checkpoint that lacks a chat template.")
    parser.add_argument("--resume-from", type=str,   default=None,
                        help="Path to a checkpoint directory to resume from, or 'latest'")
    args = parser.parse_args()

    train_ds, val_ds = load_splits(args.data, sample=args.sample)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "backward_prefetch": "backward_pre",
            "use_orig_params": True,
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "activation_checkpointing": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
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
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
