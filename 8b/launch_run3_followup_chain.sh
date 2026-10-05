#!/bin/bash
# launch_run3_followup_chain.sh — handles the two follow-up conditions:
#   1. raw_base: eval-only (no training), then GPT-4o score
#   2. sft_instruct: train (QLoRA, auto-resubmit on timeout) -> eval -> score
# Both run independently; script exits once both are fully scored. No time caps.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_run3_followup_chain.log
POLL_SECONDS=300

echo "$(date) Starting Run 3 follow-up chain monitor (PID $$), no time cap" >> "$LOGFILE"

# --- raw_base: eval only ---
raw_base_eval_done=0
raw_base_scored=0

# --- sft_instruct: train -> eval ---
sft_instruct_train_done=0
sft_instruct_eval_done=0
sft_instruct_scored=0

while true; do
    # raw_base eval
    if [ "$raw_base_eval_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_eval_run3_raw_base\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/eval_run3_raw_base_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "eval_run3_raw_base complete." "$latest"; then
                raw_base_eval_done=1
                echo "$(date) raw_base eval COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Resubmitting raw_base eval" >> "$LOGFILE"
                sbatch run_eval_run3_raw_base.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    # sft_instruct training
    if [ "$sft_instruct_train_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_sft_instruct_qlora_run3\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/sft_instruct_qlora_run3_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "SFT-from-Instruct QLoRA complete." "$latest"; then
                sft_instruct_train_done=1
                echo "$(date) sft_instruct training COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Resubmitting sft_instruct training" >> "$LOGFILE"
                sbatch run_sft_instruct_qlora_run3.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    # sft_instruct eval (only once training is done)
    if [ "$sft_instruct_train_done" == "1" ] && [ "$sft_instruct_eval_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_eval_run3_sft_instruct\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/eval_run3_sft_instruct_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "eval_run3_sft_instruct complete." "$latest"; then
                sft_instruct_eval_done=1
                echo "$(date) sft_instruct eval COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Submitting sft_instruct eval" >> "$LOGFILE"
                sbatch run_eval_run3_sft_instruct.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    # Once both evals are done, score whichever hasn't been scored yet
    if [ "$raw_base_eval_done" == "1" ] && [ "$raw_base_scored" == "0" ] && [ -f /uss/skavlak/version_3/eval_results/run3_raw_base_scored.jsonl ]; then
        n=$(wc -l < /uss/skavlak/version_3/eval_results/run3_raw_base_scored.jsonl 2>/dev/null || echo 0)
        [ "$n" -ge 800 ] && raw_base_scored=1
    fi
    if [ "$sft_instruct_eval_done" == "1" ] && [ "$sft_instruct_scored" == "0" ] && [ -f /uss/skavlak/version_3/eval_results/run3_sft_instruct_scored.jsonl ]; then
        n=$(wc -l < /uss/skavlak/version_3/eval_results/run3_sft_instruct_scored.jsonl 2>/dev/null || echo 0)
        [ "$n" -ge 800 ] && sft_instruct_scored=1
    fi

    if [ "$raw_base_eval_done" == "1" ] && [ "$raw_base_scored" == "0" ] || [ "$sft_instruct_eval_done" == "1" ] && [ "$sft_instruct_scored" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_run3_followup\$")
        if [ "$active" == "0" ]; then
            echo "$(date) Submitting follow-up scoring" >> "$LOGFILE"
            sbatch run_score_run3_followup.slurm >> "$LOGFILE" 2>&1
        fi
    fi

    if [ "$raw_base_scored" == "1" ] && [ "$sft_instruct_scored" == "1" ]; then
        echo "$(date) BOTH FOLLOW-UP CONDITIONS FULLY SCORED. Exiting monitor." >> "$LOGFILE"
        break
    fi

    sleep "$POLL_SECONDS"
done
