#!/bin/bash
#SBATCH --job-name=tb_score_8b_base_r1_rand
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_8b_base_r1_rand_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_8b_base_r1_rand_%j.err
# No --gpus: GPT-4o API scoring, no local GPU inference.

mkdir -p /home/skavlak/finetuning/logs
echo "=== scoring 8b_base_r1_rand ==="
/home/skavlak/miniconda3/envs/autorag/bin/python /home/skavlak/finetuning/data_prep/score_explanations.py \
    --results /uss/skavlak/tb_corpus_v3/eval_results/8b_base_r1_rand.jsonl \
    --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_base_r1_rand_scored.jsonl \
    --randomize-answers \
    --seed 42 \
    --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED (exit $status)"
    exit $status
fi
echo "score_8b_base_r1_rand complete."
