#!/bin/bash
#SBATCH --job-name=tb_aligned_mcq_scaled
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/aligned_mcq_scaled_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/aligned_mcq_scaled_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
MCQ=/uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results
MODEL=meta-llama/Llama-3.1-8B-Instruct

echo "=== aligned_scaled_base (no adapter) ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MODEL" --tokenizer "$MODEL" \
    --out "$OUT_DIR/aligned_scaled_base.jsonl" --randomize-answers --seed 42 --resume

echo "=== aligned_scaled_run3_sft_instruct ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MODEL" --tokenizer "$MODEL" \
    --adapter /uss/skavlak/tb_sft_instruct_run3_qlora_checkpoints/final \
    --out "$OUT_DIR/aligned_scaled_run3_sft_instruct.jsonl" --randomize-answers --seed 42 --resume

echo "=== aligned_scaled_run4_sft_instruct ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MODEL" --tokenizer "$MODEL" \
    --adapter /uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints/final \
    --out "$OUT_DIR/aligned_scaled_run4_sft_instruct.jsonl" --randomize-answers --seed 42 --resume

echo "aligned_mcq_scaled generation complete."
