#!/bin/bash
#SBATCH --job-name=tb_rag_mm_shard_full
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/rag_mm_shard_full_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/rag_mm_shard_full_%j.err

mkdir -p /home/skavlak/finetuning/logs
source /home/skavlak/miniconda3/etc/profile.d/conda.sh
conda activate /uss/skavlak/envs/jina_rag_env
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)
cd /home/skavlak/finetuning/rag

# SHARD_ID and NUM_SHARDS passed in via sbatch --export
python3 build_index_medimind_full.py --batch-size 8 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS"
status=$?

if [ $status -ne 0 ]; then
    echo "RAG MediMind (full/upper-bound) shard $SHARD_ID/$NUM_SHARDS FAILED (exit $status)"
    exit $status
fi
echo "RAG MediMind (full/upper-bound) shard $SHARD_ID/$NUM_SHARDS complete."
