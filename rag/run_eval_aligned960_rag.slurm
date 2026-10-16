#!/bin/bash
#SBATCH --job-name=tb_eval_aligned960_rag
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_aligned960_rag_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_aligned960_rag_%j.err

mkdir -p /home/skavlak/finetuning/logs
cd /home/skavlak/finetuning/rag

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
MCQ=/uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl
OUT=/uss/skavlak/version_3/eval_results/aligned960_v5_sft_instruct_100k_rag.jsonl
MODEL=meta-llama/Llama-3.1-8B-Instruct
ADAPTER=/uss/skavlak/tb_sft_instruct_v5_100k_qlora_checkpoints/final

echo "=== SFT + RAG (v5 SFT-Instruct-100k, retrieving from full v5 corpus) on aligned-960 ==="

$PYTHON evaluate_emcqa_rag.py \
    --mcq "$MCQ" --model "$MODEL" --adapter "$ADAPTER" \
    --out "$OUT" \
    --randomize-answers --seed 42 --resume --batch-size 2
status=$?

if [ $status -ne 0 ]; then
    echo "EVAL FAILED (exit $status)"
    exit $status
fi
echo "eval_aligned960_v5_sft_instruct_100k_rag complete."
