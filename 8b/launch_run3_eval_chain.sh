#!/bin/bash
# launch_run3_eval_chain.sh — watch all 3 Run 3 eval inference jobs (base,
# SFT-only, SFT-from-CPT), auto-resubmitting any that get cut off
# (evaluate_emcqa.py's --resume makes this lossless), then auto-launch
# GPT-4o scoring for all 3 once inference is fully done. No time caps.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_run3_eval_chain.log
POLL_SECONDS=300

declare -A JOBS=(
    [tb_eval_run3_base]="run_eval_run3_base.slurm|eval_run3_base|eval_run3_base complete."
    [tb_eval_run3_sft]="run_eval_run3_sft.slurm|eval_run3_sft|eval_run3_sft complete."
    [tb_eval_run3_cpt_sft]="run_eval_run3_cpt_sft.slurm|eval_run3_cpt_sft|eval_run3_cpt_sft complete."
)

echo "$(date) Starting Run 3 eval chain monitor (PID $$), 3 conditions, no time cap" >> "$LOGFILE"

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
        echo "$(date) ALL 3 RUN 3 EVAL JOBS COMPLETE. Submitting GPT-4o scoring." >> "$LOGFILE"
        break
    fi
    sleep "$POLL_SECONDS"
done

cat > run_score_run3.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=tb_score_run3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_run3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_run3_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results

for cond in run3_base run3_sft run3_cpt_sft; do
    echo "=== scoring $cond ==="
    $PYTHON "$SCRIPT" --mcq "$MCQ" --results "$OUT_DIR/${cond}.jsonl" --out "$OUT_DIR/${cond}_scored.jsonl" --randomize-answers --seed 42 --resume
    status=$?
    if [ $status -ne 0 ]; then
        echo "SCORING FAILED at $cond, exit $status"
        exit $status
    fi
done
echo "Run 3 scoring complete."
EOF

sbatch run_score_run3.slurm >> "$LOGFILE" 2>&1
sleep 15

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_run3\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest=$(ls -t /home/skavlak/finetuning/logs/score_run3_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ] && grep -q "^Run 3 scoring complete\.\$" "$latest"; then
        echo "$(date) SCORING COMPLETE (found in $latest). Run 3 fully done. Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting scoring" >> "$LOGFILE"
    sbatch run_score_run3.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done

echo "$(date) launch_run3_eval_chain.sh finished." >> "$LOGFILE"
