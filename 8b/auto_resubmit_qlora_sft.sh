#!/bin/bash
# auto_resubmit_qlora_sft.sh — keep resubmitting both QLoRA SFT jobs (SFT-only and
# CPT+SFT) until each completes. Each ~12h SLURM window only covers a fraction of
# the ~44.7h total training time these runs need (measured from the first attempt:
# 37,794 total steps, ~3.4s/step steady state). --resume-from latest in each
# script picks up from the last saved checkpoint, so resubmitting is lossless.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_qlora_sft.log
POLL_SECONDS=300

declare -A JOBS=(
    [tb_sft_qlora_run2]="run_sft_qlora_run2.slurm"
    [tb_sft_cpt_qlora_run2]="run_sft_cpt_qlora_run2.slurm"
)
declare -A done_flag
for j in "${!JOBS[@]}"; do done_flag[$j]=0; done

echo "$(date) Starting QLoRA SFT auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

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

        # Not active. Check whether it actually finished vs got cut off.
        prefix=$(echo "$jobname" | sed 's/^tb_//')
        latest_log=$(ls -t /home/skavlak/finetuning/logs/${prefix}_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ] && grep -qE "^(SFT-only QLoRA complete\.|SFT-from-CPT QLoRA complete\.)\$" "$latest_log"; then
            done_flag[$jobname]=1
            echo "$(date) $jobname COMPLETE (found in $latest_log)" >> "$LOGFILE"
            continue
        fi

        echo "$(date) Resubmitting $jobname" >> "$LOGFILE"
        sbatch "${JOBS[$jobname]}" >> "$LOGFILE" 2>&1
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) BOTH QLoRA SFT JOBS COMPLETE. Exiting monitor." >> "$LOGFILE"
        break
    fi

    sleep "$POLL_SECONDS"
done
