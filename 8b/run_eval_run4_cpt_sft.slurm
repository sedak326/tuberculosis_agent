#!/bin/bash
#SBATCH --job-name=tb_eval_run4_cpt_sft
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_run4_cpt_sft_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_run4_cpt_sft_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
ADAPTER=/uss/skavlak/tb_sft_cpt_run4_qlora_checkpoints/final

echo "=== run4_cpt_sft on main 800-Q MCQ set ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        /uss/skavlak/version_3/mcq_eval.jsonl \
    --model      meta-llama/Llama-3.1-8B \
    --adapter    "$ADAPTER" \
    --tokenizer  meta-llama/Llama-3.1-8B-Instruct \
    --out        /uss/skavlak/version_3/eval_results/run4_cpt_sft.jsonl \
    --randomize-answers --seed 42 --resume

echo "=== run4_cpt_sft on aligned (scaled) MCQ set ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        /uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl \
    --model      meta-llama/Llama-3.1-8B \
    --adapter    "$ADAPTER" \
    --tokenizer  meta-llama/Llama-3.1-8B-Instruct \
    --out        /uss/skavlak/version_3/eval_results/aligned_scaled_cpt_sft_run4.jsonl \
    --randomize-answers --seed 42 --resume

echo "eval_run4_cpt_sft complete."
