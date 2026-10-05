#!/bin/bash
#SBATCH --job-name=tb_score_run2
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_run2_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_run2_%j.err
# No --gpus: this is GPT-4o API scoring, not local inference.

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
OUT_DIR=/uss/skavlak/tb_corpus_v3/eval_results

for cond in 8b_base_raw_run2 8b_sft_qlora_run2 8b_sft_cpt_qlora_run2; do
    echo "=== scoring $cond ==="
    $PYTHON "$SCRIPT" \
        --results "$OUT_DIR/${cond}.jsonl" \
        --out     "$OUT_DIR/${cond}_scored.jsonl" \
        --resume
    status=$?
    if [ $status -ne 0 ]; then
        echo "SCORING FAILED at $cond, exit $status"
        exit $status
    fi
done

echo "Run 2 scoring complete."
