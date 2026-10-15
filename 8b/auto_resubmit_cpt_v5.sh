#!/bin/bash
# auto_resubmit_cpt_v5.sh - keep resubmitting the v5 full-parameter CPT job until
# it completes 2 full epochs over the ~173M-token combined corpus. Each 6-hour
# SLURM window times out before finishing; train_cpt.py's --resume-from latest
# picks up from the last saved checkpoint automatically (handled inside
# run_cpt_v5_full.slurm), so resubmitting is safe and lossless.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_cpt_v5.log
POLL_SECONDS=300
JOB_NAME=tb_cpt_v5_full

echo "$(date) Starting CPT v5 auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest_log=$(ls -t /home/skavlak/finetuning/logs/cpt_v5_full_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^CPT (v5, full-parameter) complete\.\$" "$latest_log"; then
        echo "$(date) CPT v5 COMPLETE (found in $latest_log). Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting CPT (v5, full-parameter)" >> "$LOGFILE"
    sbatch run_cpt_v5_full.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
