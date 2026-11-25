#!/bin/bash
#SBATCH --job-name=tb_oldeval_paper3rag_s1
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/oldeval_paper3rag_s1_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/oldeval_paper3rag_s1_%j.err

mkdir -p /home/skavlak/finetuning/logs
cd /home/skavlak/finetuning/rag
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY not set - aborting (needed for query embeddings)."
    exit 1
fi

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT=/uss/skavlak/version_3/eval_results/oldeval_v5_paper3_rag_shard1.jsonl
MODEL=meta-llama/Llama-3.1-8B-Instruct
ADAPTER=/uss/skavlak/tb_sft_instruct_v5_100k_qlora_checkpoints/final

echo "=== oldeval Bal & Puhan (paper3) RAG (SFT-3 generator) shard 1/2 on old 800-Q mcq_eval.jsonl ==="
$PYTHON evaluate_emcqa_rag_paper3.py \
    --mcq "$MCQ" --model "$MODEL" --adapter "$ADAPTER" \
    --out "$OUT" \
    --ids-file /uss/skavlak/version_3/oldeval_shard1_ids.json \
    --randomize-answers --seed 42 --resume --batch-size 2
status=$?
if [ $status -ne 0 ]; then
    echo "SHARD 1 FAILED, exit $status"
    exit $status
fi
echo "oldeval_paper3rag_shard1 complete."
