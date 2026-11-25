#!/bin/bash
# Wait for the v6 (BioMistral-recipe QLoRA) aligned-960 inference job to
# finish, then submit its GPT-4o scoring job.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/watch_eval_v6_chain.log
POLL_SECONDS=120
JOB_ID=300154

echo "$(date) Starting v6-eval -> score chain watcher (PID $$), watching job $JOB_ID" >> "$LOGFILE"

while true; do
    sleep "$POLL_SECONDS"
    state=$(sacct -j "$JOB_ID" -n -o State --format=State%20 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$state" ]; then
        state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
    fi
    echo "$(date) job $JOB_ID state: $state" >> "$LOGFILE"
    if [[ "$state" == COMPLETED* ]]; then
        echo "$(date) v6 eval inference complete." >> "$LOGFILE"
        break
    fi
    if [[ "$state" == FAILED* || "$state" == CANCELLED* || "$state" == TIMEOUT* ]]; then
        echo "$(date) v6 eval inference FAILED (state=$state). Resubmitting." >> "$LOGFILE"
        NEW_ID=$(sbatch --export=ALL,COND_NAME=v6_biomistral_qlora,MODEL=meta-llama/Llama-3.1-8B-Instruct,TOKENIZER=meta-llama/Llama-3.1-8B-Instruct,ADAPTER=/uss/skavlak/tb_sft_v6_biomistral_qlora_checkpoints/final run_eval_aligned960_generic.slurm | awk '{print $NF}')
        JOB_ID="$NEW_ID"
        echo "$(date) Resubmitted as job $JOB_ID" >> "$LOGFILE"
    fi
done

echo "$(date) Submitting v6_biomistral_qlora scoring job" >> "$LOGFILE"
sbatch run_score_aligned960_v6_biomistral.slurm >> "$LOGFILE" 2>&1
echo "$(date) Chain watcher done." >> "$LOGFILE"
