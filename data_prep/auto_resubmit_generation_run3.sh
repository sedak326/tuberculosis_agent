#!/bin/bash
# auto_resubmit_generation_run3.sh — keep resubmitting the Run 3 SFT generation
# job until it completes (24,000-chunk sample -> ~40K examples). Single job,
# no array needed (see run_generation_run3.slurm). generate_training_data.py
# skips already-processed chunk IDs automatically by reading the output file,
# so resubmitting on SLURM timeout is lossless.

cd /home/skavlak/finetuning/data_prep
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_generation_run3.log
POLL_SECONDS=300
JOB_NAME=tb_data_gen_run3

echo "$(date) Starting Run 3 SFT generation auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest_log=$(ls -t /home/skavlak/finetuning/logs/generation_run3_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^generation_run3 complete\.\$" "$latest_log"; then
        echo "$(date) GENERATION COMPLETE (found in $latest_log). Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting generation (Run 3)" >> "$LOGFILE"
    sbatch run_generation_run3.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
