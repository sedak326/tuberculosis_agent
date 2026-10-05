#!/bin/bash
# Generates 9 individual GPT-4o scoring SLURM scripts (one per randomized-eval
# condition) so they run in parallel instead of sequentially in one job.
# No GPU requested — scoring is OpenAI API calls, CPU/network-bound only.
set -e
OUT=/home/skavlak/finetuning/8b
LOG=/home/skavlak/finetuning/logs
RESULTS=/uss/skavlak/tb_corpus_v3/eval_results
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py

CONDS="8b_base_r1_rand 8b_sft_r1_rand 8b_cpt_sft_r1_rand 70b_base_r1_rand 70b_instruct_r1_rand 70b_sft_r1_rand 8b_base_r2_rand 8b_sft_qlora_r2_rand 8b_sft_cpt_qlora_r2_rand"

for cond in $CONDS; do
  cat > "$OUT/run_score_${cond}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=tb_score_${cond}
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG}/score_${cond}_%j.log
#SBATCH --error=${LOG}/score_${cond}_%j.err
# No --gpus: GPT-4o API scoring, no local GPU inference.

mkdir -p ${LOG}
echo "=== scoring ${cond} ==="
${PYTHON} ${SCRIPT} \\
    --results ${RESULTS}/${cond}.jsonl \\
    --out     ${RESULTS}/${cond}_scored.jsonl \\
    --randomize-answers \\
    --seed 42 \\
    --resume
status=\$?
if [ \$status -ne 0 ]; then
    echo "SCORING FAILED (exit \$status)"
    exit \$status
fi
echo "score_${cond} complete."
EOF
  echo "wrote run_score_${cond}.slurm"
done
