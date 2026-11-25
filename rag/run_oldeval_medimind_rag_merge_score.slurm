#!/bin/bash
#SBATCH --job-name=tb_oldeval_medimindrag_merge_score
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/oldeval_medimindrag_merge_score_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/oldeval_medimindrag_merge_score_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
RAW=/uss/skavlak/version_3/eval_results/oldeval_v5_medimind_rag.jsonl
SCORED=/uss/skavlak/version_3/eval_results/oldeval_v5_medimind_rag_scored.jsonl

echo "=== merging shards for medimind RAG ==="
cat "/uss/skavlak/version_3/eval_results/oldeval_v5_medimind_rag_shard0.jsonl" "/uss/skavlak/version_3/eval_results/oldeval_v5_medimind_rag_shard1.jsonl" > "$RAW"
n=$(wc -l < "$RAW")
echo "merged $n lines into $RAW"

echo "=== scoring with GPT-4o ==="
$PYTHON /home/skavlak/finetuning/data_prep/score_explanations.py \
    --mcq "/uss/skavlak/version_3/mcq_eval.jsonl" \
    --results "$RAW" \
    --out "$SCORED" \
    --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED, exit $status"
    exit $status
fi
echo "oldeval_medimind_rag_merge_score complete."
