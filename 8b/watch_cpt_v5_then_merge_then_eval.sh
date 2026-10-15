#!/bin/bash
# Chain: wait for CPT v5 to finish -> submit SLERP merge -> wait for merge to
# finish -> submit eval of merged-no-SFT on the aligned-960 MCQA set -> drop a
# marker file so the SFT-launch watcher knows the merge is ready.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/watch_cpt_v5_chain.log
POLL_SECONDS=300
MERGED=/uss/skavlak/tb_cpt_v5_full_instruct_slerp_merged
MARKER=/uss/skavlak/version_5/.merge_ready

echo "$(date) Starting CPT v5 -> merge -> eval chain watcher (PID $$)" >> "$LOGFILE"

# --- Step 1: wait for CPT v5 completion ---
while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_cpt_v5_full\$")
    latest_log=$(ls -t /home/skavlak/finetuning/logs/cpt_v5_full_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^CPT (v5, full-parameter) complete\.\$" "$latest_log"; then
        echo "$(date) CPT v5 complete." >> "$LOGFILE"
        break
    fi
    if [ "$active" -eq 0 ]; then
        echo "$(date) WARNING: no CPT v5 job active and no completion marker yet - relying on auto_resubmit_cpt_v5.sh to resubmit." >> "$LOGFILE"
    fi
    sleep "$POLL_SECONDS"
done

# --- Step 2: submit merge, wait for it ---
echo "$(date) Submitting SLERP merge" >> "$LOGFILE"
sbatch run_merge_v5_cpt_instruct.slurm >> "$LOGFILE" 2>&1
while true; do
    sleep "$POLL_SECONDS"
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_merge_v5_cpt_instruct\$")
    latest_log=$(ls -t /home/skavlak/finetuning/logs/merge_v5_cpt_instruct_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^merge_v5_cpt_instruct complete\." "$latest_log"; then
        echo "$(date) Merge complete." >> "$LOGFILE"
        break
    fi
    if [ "$active" -eq 0 ] && [ -n "$latest_log" ] && grep -qi "FAILED" "$latest_log"; then
        echo "$(date) MERGE FAILED, see $latest_log. Exiting chain." >> "$LOGFILE"
        exit 1
    fi
done

# --- Step 3: submit merged-no-SFT eval on aligned-960 set ---
echo "$(date) Submitting merged-no-SFT aligned eval" >> "$LOGFILE"
sbatch --export=ALL,COND_NAME=v5_merged_nosft,MODEL="$MERGED",TOKENIZER="$MERGED" run_eval_aligned960_generic.slurm >> "$LOGFILE" 2>&1

touch "$MARKER"
echo "$(date) Merge ready marker written to $MARKER. Chain watcher done." >> "$LOGFILE"
