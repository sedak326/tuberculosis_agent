#!/bin/bash
# Wait for BOTH the v5 merge and the finalized 100k SFT data to be ready, then
# submit SFT on the merged model, and once that finishes, submit the eval of
# merged+SFT100k on the aligned-960 MCQA set.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/watch_sft_v5_chain.log
POLL_SECONDS=300
MERGED=/uss/skavlak/tb_cpt_v5_full_instruct_slerp_merged
ADAPTER=/uss/skavlak/tb_sft_v5_merged_100k_qlora_checkpoints/final
MERGE_MARKER=/uss/skavlak/version_5/.merge_ready
DATA_MARKER=/uss/skavlak/version_5/.100k_data_ready

echo "$(date) Starting merge+100k-data -> SFT -> eval chain watcher (PID $$)" >> "$LOGFILE"

while true; do
    if [ -f "$MERGE_MARKER" ] && [ -f "$DATA_MARKER" ]; then
        echo "$(date) Both merge and 100k data ready." >> "$LOGFILE"
        break
    fi
    sleep "$POLL_SECONDS"
done

echo "$(date) Submitting SFT (QLoRA, 100k data) on v5 merged model" >> "$LOGFILE"
sbatch run_sft_qlora_v5_merged_100k.slurm >> "$LOGFILE" 2>&1

while true; do
    sleep "$POLL_SECONDS"
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_sft_qlora_v5_merged_100k\$")
    latest_log=$(ls -t /home/skavlak/finetuning/logs/sft_qlora_v5_merged_100k_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^SFT QLoRA on v5 merged model complete\.\$" "$latest_log"; then
        echo "$(date) SFT complete." >> "$LOGFILE"
        break
    fi
    if [ "$active" -eq 0 ] && [ -n "$latest_log" ] && grep -qi "FAILED" "$latest_log"; then
        echo "$(date) SFT FAILED, see $latest_log. Resubmitting (train_sft.py QLoRA checkpoints are resumable)." >> "$LOGFILE"
        sbatch run_sft_qlora_v5_merged_100k.slurm >> "$LOGFILE" 2>&1
    fi
done

echo "$(date) Submitting merged+SFT100k aligned eval" >> "$LOGFILE"
sbatch --export=ALL,COND_NAME=v5_merged_sft100k,MODEL="$MERGED",TOKENIZER="$MERGED",ADAPTER="$ADAPTER" run_eval_aligned960_generic.slurm >> "$LOGFILE" 2>&1
echo "$(date) Chain watcher done." >> "$LOGFILE"
