#!/bin/bash
#SBATCH --job-name=tb_rag_build_index
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/rag_build_index_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/rag_build_index_%j.err

mkdir -p /home/skavlak/finetuning/logs
cd /home/skavlak/finetuning/rag

export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

/home/skavlak/miniconda3/envs/autorag/bin/python3 build_index.py
status=$?

if [ $status -ne 0 ]; then
    echo "RAG index build FAILED (exit $status)"
    exit $status
fi
echo "tb_rag_build_index complete."
