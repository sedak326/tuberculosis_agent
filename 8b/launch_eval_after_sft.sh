#!/bin/bash
# launch_eval_after_sft.sh — full Run 2 eval chain, unattended:
#   1. Wait for both SFT QLoRA jobs (SFT-only, CPT+SFT) to genuinely finish.
#   2. Submit all 3 MCQ eval inference jobs in parallel (1 GPU each — plenty of
#      GPUs free once the SFT jobs release theirs).
#   3. Wait for all 3 to finish, auto-resubmitting any that get cut off
#      (evaluate_emcqa.py's --resume makes this lossless).
#   4. Submit GPT-4o scoring (API-based, no GPU) for all 3 result files.
#   5. Wait for scoring to finish, auto-resubmitting if needed
#      (score_explanations.py's --resume makes this lossless too).
# No time caps anywhere in this chain — polls rather than depending on any one
# SLURM job ID, since jobs may need further resubmission cycles.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_eval.log
POLL_SECONDS=300

echo "$(date) Watching for both SFT jobs to complete (PID $$)" >> "$LOGFILE"

sft_only_done=0
cpt_sft_done=0
while true; do
    if [ "$sft_only_done" == "0" ]; then
        latest=$(ls -t /home/skavlak/finetuning/logs/sft_qlora_run2_*.log 2>/dev/null | head -1)
        [ -n "$latest" ] && grep -q "^SFT-only QLoRA complete\.\$" "$latest" && sft_only_done=1 && \
            echo "$(date) SFT-only QLoRA confirmed complete" >> "$LOGFILE"
    fi
    if [ "$cpt_sft_done" == "0" ]; then
        latest=$(ls -t /home/skavlak/finetuning/logs/sft_cpt_qlora_run2_*.log 2>/dev/null | head -1)
        [ -n "$latest" ] && grep -q "^SFT-from-CPT QLoRA complete\.\$" "$latest" && cpt_sft_done=1 && \
            echo "$(date) CPT+SFT QLoRA confirmed complete" >> "$LOGFILE"
    fi
    [ "$sft_only_done" == "1" ] && [ "$cpt_sft_done" == "1" ] && break
    sleep "$POLL_SECONDS"
done

echo "$(date) Both SFT jobs complete. Submitting 3 parallel eval jobs." >> "$LOGFILE"

declare -A EVAL_JOBS=(
    [tb_eval_base_run2]="run_eval_base_run2.slurm|eval_base_run2|eval_base_run2 complete."
    [tb_eval_sft_run2]="run_eval_sft_run2.slurm|eval_sft_run2|eval_sft_run2 complete."
    [tb_eval_cpt_sft_run2]="run_eval_cpt_sft_run2.slurm|eval_cpt_sft_run2|eval_cpt_sft_run2 complete."
)
for jobname in "${!EVAL_JOBS[@]}"; do
    IFS='|' read -r script _ _ <<< "${EVAL_JOBS[$jobname]}"
    sbatch "$script" >> "$LOGFILE" 2>&1
done
sleep 30  # let SLURM register all 3 before the watch loop checks them

declare -A eval_done
for jobname in "${!EVAL_JOBS[@]}"; do eval_done[$jobname]=0; done

while true; do
    all_done=1
    for jobname in "${!EVAL_JOBS[@]}"; do
        if [ "${eval_done[$jobname]}" == "1" ]; then
            continue
        fi
        all_done=0

        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^${jobname}\$")
        if [ "$active" -gt 0 ]; then
            continue
        fi

        IFS='|' read -r script logprefix donemsg <<< "${EVAL_JOBS[$jobname]}"
        latest=$(ls -t /home/skavlak/finetuning/logs/${logprefix}_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ] && grep -qF "$donemsg" "$latest"; then
            eval_done[$jobname]=1
            echo "$(date) $jobname COMPLETE (found in $latest)" >> "$LOGFILE"
            continue
        fi

        echo "$(date) Resubmitting $jobname" >> "$LOGFILE"
        sbatch "$script" >> "$LOGFILE" 2>&1
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) ALL 3 EVAL JOBS COMPLETE. Submitting GPT-4o scoring." >> "$LOGFILE"
        break
    fi
    sleep "$POLL_SECONDS"
done

sbatch run_score_run2.slurm >> "$LOGFILE" 2>&1
sleep 15

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_run2\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    latest=$(ls -t /home/skavlak/finetuning/logs/score_run2_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ] && grep -q "^Run 2 scoring complete\.\$" "$latest"; then
        echo "$(date) SCORING COMPLETE (found in $latest). Run 2 fully done. Exiting monitor." >> "$LOGFILE"
        break
    fi

    echo "$(date) Resubmitting scoring" >> "$LOGFILE"
    sbatch run_score_run2.slurm >> "$LOGFILE" 2>&1
    sleep "$POLL_SECONDS"
done

echo "$(date) launch_eval_after_sft.sh finished." >> "$LOGFILE"
