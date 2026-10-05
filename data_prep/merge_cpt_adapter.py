#!/usr/bin/env python3
"""
merge_cpt_adapter.py — Merge a QLoRA CPT adapter into the base model, producing
a real full-precision checkpoint that a subsequent SFT-QLoRA stage can build on.

Necessary because train_sft.py has no --adapter flag (it can only start from a
plain model ID/path) — Run 2's "SFT from CPT checkpoint" pattern assumed
full-parameter CPT output (already a full model). Run 3's CPT is QLoRA, so its
checkpoint is a LoRA adapter only; this bridges the gap the same way any
LoRA-CPT-then-LoRA-SFT pipeline needs to: merge the first adapter's delta into
the base weights in full precision, save that as a new "base", then the SFT
stage re-quantizes THAT for its own fresh LoRA adapter.

Usage:
    python merge_cpt_adapter.py --base meta-llama/Llama-3.1-8B \
        --adapter /uss/skavlak/tb_cpt_run3_qlora_checkpoints/final \
        --out /uss/skavlak/tb_cpt_run3_qlora_merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading base model {args.base} in bf16...")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")

    print(f"Loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.out}...")
    merged.save_pretrained(args.out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.out)

    print("Done.")


if __name__ == "__main__":
    main()
