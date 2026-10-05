#!/bin/bash
#SBATCH --job-name=tb_score_rand
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_rand_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_rand_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
OUT_DIR=/uss/skavlak/tb_corpus_v3/eval_results

for cond in 8b_base_r1_rand 8b_sft_r1_rand 8b_cpt_sft_r1_rand 70b_base_r1_rand 70b_instruct_r1_rand 70b_sft_r1_rand 8b_base_r2_rand 8b_sft_qlora_r2_rand 8b_sft_cpt_qlora_r2_rand; do
    echo "=== scoring $cond ==="
    $PYTHON "$SCRIPT" --results "$OUT_DIR/${cond}.jsonl" --out "$OUT_DIR/${cond}_scored.jsonl" --randomize-answers --seed 42 --resume
    status=$?
    if [ $status -ne 0 ]; then
        echo "SCORING FAILED at $cond, exit $status"
        exit $status
    fi
done
echo "Randomized-eval scoring complete."
