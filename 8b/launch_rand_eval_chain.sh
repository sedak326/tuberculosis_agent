#!/bin/bash
# launch_rand_eval_chain.sh — watch all 9 randomized-answer re-eval jobs (6 Run 1
# non-MIWV conditions + 3 Run 2 conditions), auto-resubmitting any that get cut
# off (evaluate_emcqa.py's --resume makes this lossless), then auto-launch
# GPT-4o scoring for all 9 once inference is fully done. No time caps anywhere.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_rand_eval_chain.log
POLL_SECONDS=300

declare -A JOBS=(
    [tb_eval_8b_base_r1]="run_eval_8b_base_r1.slurm|eval_8b_base_r1|eval_8b_base_r1 complete."
    [tb_eval_8b_sft_r1]="run_eval_8b_sft_r1.slurm|eval_8b_sft_r1|eval_8b_sft_r1 complete."
    [tb_eval_8b_cpt_sft_r1]="run_eval_8b_cpt_sft_r1.slurm|eval_8b_cpt_sft_r1|eval_8b_cpt_sft_r1 complete."
    [tb_eval_70b_base_r1]="run_eval_70b_base_r1.slurm|eval_70b_base_r1|eval_70b_base_r1 complete."
    [tb_eval_70b_instruct_r1]="run_eval_70b_instruct_r1.slurm|eval_70b_instruct_r1|eval_70b_instruct_r1 complete."
    [tb_eval_70b_sft_r1]="run_eval_70b_sft_r1.slurm|eval_70b_sft_r1|eval_70b_sft_r1 complete."
    [tb_eval_8b_base_r2]="run_eval_8b_base_r2.slurm|eval_8b_base_r2|eval_8b_base_r2 complete."
    [tb_eval_8b_sft_qlora_r2]="run_eval_8b_sft_qlora_r2.slurm|eval_8b_sft_qlora_r2|eval_8b_sft_qlora_r2 complete."
    [tb_eval_8b_sft_cpt_qlora_r2]="run_eval_8b_sft_cpt_qlora_r2.slurm|eval_8b_sft_cpt_qlora_r2|eval_8b_sft_cpt_qlora_r2 complete."
)

echo "$(date) Starting randomized-eval chain monitor (PID $$), 9 conditions, no time cap" >> "$LOGFILE"

declare -A done_flag
for j in "${!JOBS[@]}"; do done_flag[$j]=0; done

while true; do
    all_done=1
    for jobname in "${!JOBS[@]}"; do
        if [ "${done_flag[$jobname]}" == "1" ]; then
            continue
        fi
        all_done=0

        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${jobname}\$")
        if [ "$active" -gt 0 ]; then
            continue
        fi

        IFS='|' read -r script logprefix donemsg <<< "${JOBS[$jobname]}"
        latest=$(ls -t /home/skavlak/finetuning/logs/${logprefix}_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ] && grep -qF "$donemsg" "$latest"; then
            done_flag[$jobname]=1
            echo "$(date) $jobname COMPLETE (found in $latest)" >> "$LOGFILE"
            continue
        fi

        echo "$(date) Resubmitting $jobname" >> "$LOGFILE"
        sbatch "$script" >> "$LOGFILE" 2>&1
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) ALL 9 RE-EVAL JOBS COMPLETE. Submitting GPT-4o scoring." >> "$LOGFILE"
        break
    fi
    sleep "$POLL_SECONDS"
done

# Build the scoring job on the fly (mirrors run_score_run2.slurm but for all 9
# _rand result files).
cat > run_score_rand.slurm <<'EOF'
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
EOF

sbatch run_score_rand.slurm >> "$LOGFILE" 2>&1
sleep 15

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_rand\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest=$(ls -t /home/skavlak/finetuning/logs/score_rand_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ] && grep -q "^Randomized-eval scoring complete\.\$" "$latest"; then
        echo "$(date) SCORING COMPLETE (found in $latest). Fully done. Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting scoring" >> "$LOGFILE"
    sbatch run_score_rand.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done

echo "$(date) launch_rand_eval_chain.sh finished." >> "$LOGFILE"
