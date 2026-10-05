#!/bin/bash
#SBATCH --job-name=tb_score_run3_followup
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_run3_followup_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_run3_followup_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results

for cond in run3_raw_base run3_sft_instruct; do
    if [ ! -f "$OUT_DIR/${cond}.jsonl" ]; then
        echo "=== $cond eval not ready yet, skipping this pass ==="
        continue
    fi
    n=$(wc -l < "$OUT_DIR/${cond}.jsonl")
    if [ "$n" -lt 800 ]; then
        echo "=== $cond eval incomplete ($n/800), skipping this pass ==="
        continue
    fi
    echo "=== scoring $cond ==="
    $PYTHON "$SCRIPT" --mcq "$MCQ" --results "$OUT_DIR/${cond}.jsonl" --out "$OUT_DIR/${cond}_scored.jsonl" --randomize-answers --seed 42 --resume
    status=$?
    if [ $status -ne 0 ]; then
        echo "SCORING FAILED at $cond, exit $status"
        exit $status
    fi
done
echo "Run 3 follow-up scoring pass complete."
