#!/bin/bash
# auto_resubmit_sft_cpt_qlora_run3.sh — keep resubmitting Run 3's SFT-from-CPT
# QLoRA job until it completes. --resume-from latest picks up from the last
# saved checkpoint automatically, so resubmitting on SLURM timeout is lossless.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_sft_cpt_qlora_run3.log
POLL_SECONDS=300
JOB_NAME=tb_sft_cpt_qlora_run3

echo "$(date) Starting Run 3 SFT-from-CPT auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest_log=$(ls -t /home/skavlak/finetuning/logs/sft_cpt_qlora_run3_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^SFT-from-CPT QLoRA complete\.\$" "$latest_log"; then
        echo "$(date) SFT-FROM-CPT COMPLETE (found in $latest_log). Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting SFT-from-CPT (Run 3)" >> "$LOGFILE"
    sbatch run_sft_cpt_qlora_run3.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done
