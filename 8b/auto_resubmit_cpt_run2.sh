#!/bin/bash
# auto_resubmit_cpt_run2.sh — keep resubmitting the Run 2 CPT job (single job, not
# sharded) until it completes 2 full epochs. Each 6-hour SLURM window times out
# before finishing on the expanded corpus; train_cpt.py's --resume-from latest
# picks up from the last saved checkpoint automatically (handled inside
# run_cpt_run2.slurm), so resubmitting is safe and lossless.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_cpt.log
POLL_SECONDS=300
JOB_NAME=tb_cpt_run2

echo "$(date) Starting CPT auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    # Not active. Check whether the last run actually finished ("CPT complete.")
    # vs got cut off (TIMEOUT/FAILED) and needs resubmission.
    latest_log=$(ls -t /home/skavlak/finetuning/logs/cpt_run2_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^CPT complete\.\$" "$latest_log"; then
        echo "$(date) CPT COMPLETE (found in $latest_log). Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting CPT (Run 2)" >> "$LOGFILE"
    sbatch run_cpt_run2.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
