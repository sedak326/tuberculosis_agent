#!/bin/bash
#SBATCH --job-name=tb_eval_run4_merged_smoothed_main
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_run4_merged_smoothed_main_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_run4_merged_smoothed_main_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
MERGED=/uss/skavlak/tb_cpt_run4_full_instruct_slerp_smoothed_merged
ADAPTER=/uss/skavlak/tb_sft_run4_merged_smoothed_qlora_checkpoints/final
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl

echo "=== smoothed merge, NO SFT - main 800-Q MCQ ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MERGED" --tokenizer "$MERGED" \
    --out /uss/skavlak/version_3/eval_results/run4_merged_smoothed_nosft.jsonl \
    --randomize-answers --seed 42 --resume

echo "=== smoothed merge + SFT - main 800-Q MCQ ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MERGED" --adapter "$ADAPTER" --tokenizer "$MERGED" \
    --out /uss/skavlak/version_3/eval_results/run4_merged_smoothed_sft.jsonl \
    --randomize-answers --seed 42 --resume

echo "eval_run4_merged_smoothed_main complete."
