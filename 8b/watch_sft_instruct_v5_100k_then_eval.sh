#!/bin/bash
# Wait for the plain SFT-from-Instruct (v5, 100k data, no merge) job to
# finish, then submit its eval on the aligned-960 MCQA set.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/watch_sft_instruct_v5_100k_chain.log
POLL_SECONDS=300
ADAPTER=/uss/skavlak/tb_sft_instruct_v5_100k_qlora_checkpoints/final

echo "$(date) Starting SFT-instruct-v5-100k -> eval chain watcher (PID $$)" >> "$LOGFILE"

while true; do
    sleep "$POLL_SECONDS"
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_sft_instruct_qlora_v5_100k\$")
    latest_log=$(ls -t /home/skavlak/finetuning/logs/sft_instruct_qlora_v5_100k_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^SFT-from-Instruct QLoRA (v5, 100k) complete\." "$latest_log"; then
        echo "$(date) SFT complete." >> "$LOGFILE"
        break
    fi
    if [ "$active" -eq 0 ] && [ -n "$latest_log" ] && grep -qi "FAILED" "$latest_log"; then
        echo "$(date) SFT FAILED, see $latest_log. Resubmitting." >> "$LOGFILE"
        sbatch run_sft_instruct_qlora_v5_100k.slurm >> "$LOGFILE" 2>&1
    fi
done

echo "$(date) Submitting SFT-instruct-v5-100k aligned eval" >> "$LOGFILE"
sbatch --export=ALL,COND_NAME=v5_sft_instruct_100k,MODEL=meta-llama/Llama-3.1-8B-Instruct,TOKENIZER=meta-llama/Llama-3.1-8B-Instruct,ADAPTER="$ADAPTER" run_eval_aligned960_generic.slurm >> "$LOGFILE" 2>&1
echo "$(date) Chain watcher done." >> "$LOGFILE"
