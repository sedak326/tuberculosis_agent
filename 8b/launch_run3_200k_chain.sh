#!/bin/bash
# launch_run3_200k_chain.sh — train SFT-from-Instruct on the full 200K dataset
# (auto-resubmit on timeout), then run logprob eval (zero OpenAI cost) on both
# that checkpoint and the Instruct baseline. No GPT-4o scoring here by design
# — this is the cheap first pass; a real scoring run is a separate decision.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/launch_run3_200k_chain.log
POLL_SECONDS=300

echo "$(date) Starting Run 3 200K-data chain monitor (PID $$), no time cap" >> "$LOGFILE"

base_logprob_done=0
train_done=0
sft_logprob_done=0

while true; do
    # base logprob (independent, can run any time)
    if [ "$base_logprob_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_logprob_run3_base\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/logprob_run3_base_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "logprob_run3_base complete." "$latest"; then
                base_logprob_done=1
                echo "$(date) base logprob COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Resubmitting base logprob" >> "$LOGFILE"
                sbatch run_logprob_run3_base.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    # 200k training
    if [ "$train_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_sft_instruct_200k_run3\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/sft_instruct_200k_run3_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "SFT-from-Instruct-200K QLoRA complete." "$latest"; then
                train_done=1
                echo "$(date) 200k training COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Resubmitting 200k training" >> "$LOGFILE"
                sbatch run_sft_instruct_200k_run3.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    # sft_instruct_200k logprob (only once training is done)
    if [ "$train_done" == "1" ] && [ "$sft_logprob_done" == "0" ]; then
        active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_logprob_run3_sft_instruct_200k\$")
        if [ "$active" == "0" ]; then
            latest=$(ls -t /home/skavlak/finetuning/logs/logprob_run3_sft_instruct_200k_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ] && grep -qF "logprob_run3_sft_instruct_200k complete." "$latest"; then
                sft_logprob_done=1
                echo "$(date) 200k logprob COMPLETE" >> "$LOGFILE"
            else
                echo "$(date) Submitting 200k logprob" >> "$LOGFILE"
                sbatch run_logprob_run3_sft_instruct_200k.slurm >> "$LOGFILE" 2>&1
            fi
        fi
    fi

    if [ "$base_logprob_done" == "1" ] && [ "$sft_logprob_done" == "1" ]; then
        echo "$(date) BOTH LOGPROB EVALS COMPLETE. Exiting monitor." >> "$LOGFILE"
        break
    fi

    sleep "$POLL_SECONDS"
done
