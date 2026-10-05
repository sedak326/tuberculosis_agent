#!/bin/bash
# auto_resubmit_cpt_run3.sh — keep resubmitting the Run 3 CPT job (QLoRA,
# post-cutoff-only corpus) until it completes 2 full epochs. Mirrors
# auto_resubmit_cpt_run2.sh: train_cpt.py's --resume-from latest picks up
# from the last saved checkpoint automatically (handled inside
# run_cpt_run3_qlora.slurm), so resubmitting on SLURM timeout is lossless.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_cpt_run3.log
POLL_SECONDS=300
JOB_NAME=tb_cpt_run3_qlora

echo "$(date) Starting Run 3 CPT auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest_log=$(ls -t /home/skavlak/finetuning/logs/cpt_run3_qlora_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^CPT complete\.\$" "$latest_log"; then
        echo "$(date) CPT COMPLETE (found in $latest_log). Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting CPT (Run 3)" >> "$LOGFILE"
    sbatch run_cpt_run3_qlora.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
