#!/bin/bash
#SBATCH --job-name=tb_score_run4
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_run4_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_run4_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results

echo "=== scoring run4_sft_instruct ==="
$PYTHON "$SCRIPT" --mcq "$MCQ" --results "$OUT_DIR/run4_sft_instruct.jsonl" --out "$OUT_DIR/run4_sft_instruct_scored.jsonl" --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED, exit $status"
    exit $status
fi
echo "Run 4 scoring complete."
