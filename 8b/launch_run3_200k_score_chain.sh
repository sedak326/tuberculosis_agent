#!/bin/bash
# launch_run3_200k_score_chain.sh — generation (evaluate_emcqa.py, local GPU
# only) then GPT-4o scoring for the SFT-from-Instruct-200K condition. No time
# caps; auto-resubmits on SLURM timeout.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_run3_200k_score_chain.log
POLL_SECONDS=300

echo "$(date) Starting Run 3 200K scoring chain monitor (PID $$), no time cap" >> "$LOGFILE"

eval_done=0
while [ "$eval_done" == "0" ]; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_eval_run3_sft_instruct_200k\$")
    if [ "$active" == "0" ]; then
        latest=$(ls -t /home/skavlak/finetuning/logs/eval_run3_sft_instruct_200k_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ] && grep -qF "eval_run3_sft_instruct_200k complete." "$latest"; then
            eval_done=1
            echo "$(date) eval COMPLETE" >> "$LOGFILE"
        else
            echo "$(date) Resubmitting eval" >> "$LOGFILE"
            sbatch run_eval_run3_sft_instruct_200k.slurm >> "$LOGFILE" 2>&1
        fi
    fi
    [ "$eval_done" == "0" ] && sleep "$POLL_SECONDS"
done

cat > run_score_run3_200k.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=tb_score_run3_200k
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/score_run3_200k_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/score_run3_200k_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/venv/bin/python3
SCRIPT=/home/skavlak/finetuning/data_prep/score_explanations.py
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results

echo "=== scoring run3_sft_instruct_200k ==="
$PYTHON "$SCRIPT" --mcq "$MCQ" --results "$OUT_DIR/run3_sft_instruct_200k.jsonl" --out "$OUT_DIR/run3_sft_instruct_200k_scored.jsonl" --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED, exit $status"
    exit $status
fi
echo "Run 3 200K scoring complete."
EOF

sbatch run_score_run3_200k.slurm >> "$LOGFILE" 2>&1
sleep 15

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_run3_200k\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi
    latest=$(ls -t /home/skavlak/finetuning/logs/score_run3_200k_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ] && grep -q "^Run 3 200K scoring complete\.\$" "$latest"; then
        echo "$(date) SCORING COMPLETE. Exiting monitor." >> "$LOGFILE"
        break
    fi
    echo "$(date) Resubmitting scoring" >> "$LOGFILE"
    sbatch run_score_run3_200k.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
