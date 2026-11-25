#!/bin/bash
#SBATCH --job-name=tb_score_v5_paper3_rag_full
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_v5_paper3_rag_full_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_v5_paper3_rag_full_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
MCQ=/uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results

echo "=== scoring v5_paper3_rag_full (aligned-960, full/upper-bound) ==="
$PYTHON "$SCRIPT" --mcq "$MCQ" \
    --results "$OUT_DIR/aligned960_v5_paper3_rag_full.jsonl" \
    --out "$OUT_DIR/aligned960_v5_paper3_rag_full_scored.jsonl" \
    --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED, exit $status"
    exit $status
fi
echo "v5_paper3_rag_full scoring complete."
