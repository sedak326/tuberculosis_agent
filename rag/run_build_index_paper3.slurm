#!/bin/bash
#SBATCH --job-name=tb_rag_build_index_paper3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/rag_build_index_paper3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/rag_build_index_paper3_%j.err

mkdir -p /home/skavlak/finetuning/logs
cd /home/skavlak/finetuning/rag

# OPENAI_API_KEY is inherited from the submitting shell's environment, same
# as the GPT-4o scoring jobs (score_explanations.py) rely on - sbatch passes
# the submitter's env through by default.
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY not set in the submitting shell's environment - aborting."
    exit 1
fi

/home/skavlak/miniconda3/envs/autorag/bin/python3 build_index_paper3.py
status=$?

if [ $status -ne 0 ]; then
    echo "RAG index build (paper3) FAILED (exit $status)"
    exit $status
fi
echo "tb_rag_build_index_paper3 complete."
