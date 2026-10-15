#!/bin/bash
#SBATCH --job-name=tb_rag_smoke_test
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/rag_smoke_test_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/rag_smoke_test_%j.err

mkdir -p /home/skavlak/finetuning/logs
cd /home/skavlak/finetuning/rag

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3

echo "=== Test 1: TB-specific question (should retrieve + ground) ==="
$PYTHON chat_rag.py --show-context \
    --question "What is the role of ESAT-6 in Mycobacterium tuberculosis virulence?"

echo "=== Test 2: off-corpus question (should fall back to general knowledge) ==="
$PYTHON chat_rag.py --show-context \
    --question "What is the capital of France?"

echo "tb_rag_smoke_test complete."
