#!/bin/bash
# launch_sft_cpt_qlora_after_cpt.sh — wait for the Run 2 CPT job to genuinely finish
# ("CPT complete." in its log, not just a timeout/OOM/drain-induced exit), then
# submit the CPT+SFT QLoRA job. Safe to leave running unattended; polls rather
# than depending on one specific SLURM job ID, since CPT may need further
# resubmission cycles (as it already has) that a --dependency=afterok wouldn't
# survive.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_sft_cpt_qlora.log
POLL_SECONDS=300

echo "$(date) Watching for CPT completion (PID $$)" >> "$LOGFILE"

while true; do
    latest_log=$(ls -t /home/skavlak/finetuning/logs/cpt_run2_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^CPT complete\.\$" "$latest_log"; then
        echo "$(date) CPT complete (found in $latest_log). Submitting SFT-from-CPT QLoRA." >> "$LOGFILE"
        sbatch run_sft_cpt_qlora_run2.slurm >> "$LOGFILE" 2>&1
        break
    fi
    sleep "$POLL_SECONDS"
done

echo "$(date) launch_sft_cpt_qlora_after_cpt.sh finished." >> "$LOGFILE"
