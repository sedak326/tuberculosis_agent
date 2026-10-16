#!/bin/bash
# Wait for the SFT+RAG aligned-960 eval to finish, then submit GPT-4o scoring
# (Batch API, via score_explanations.py) automatically.

cd /home/skavlak/finetuning/data_prep
LOGFILE=/home/skavlak/finetuning/logs/watch_rag_eval_chain.log
POLL_SECONDS=300
OUT=/uss/skavlak/version_3/eval_results/aligned960_v5_sft_instruct_100k_rag.jsonl
SCORED=/uss/skavlak/version_3/eval_results/aligned960_v5_sft_instruct_100k_rag_scored.jsonl

echo "$(date) Starting SFT+RAG eval -> scoring watcher (PID $$)" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_eval_aligned960_rag\$")
    latest_log=$(ls -t /home/skavlak/finetuning/logs/eval_aligned960_rag_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ] && grep -q "^eval_aligned960_v5_sft_instruct_100k_rag complete\." "$latest_log"; then
        echo "$(date) Eval complete." >> "$LOGFILE"
        break
    fi
    if [ "$active" -eq 0 ] && [ -n "$latest_log" ] && grep -qi "FAILED" "$latest_log"; then
        echo "$(date) EVAL FAILED, see $latest_log. Resubmitting (resumable via --resume)." >> "$LOGFILE"
        cd /home/skavlak/finetuning/rag
        sbatch run_eval_aligned960_rag.slurm >> "$LOGFILE" 2>&1
        cd /home/skavlak/finetuning/data_prep
    fi
    sleep "$POLL_SECONDS"
done

n=$(wc -l < "$OUT")
echo "$(date) $n/960 lines in $OUT. Submitting GPT-4o scoring (batch mode)." >> "$LOGFILE"

/home/skavlak/venv/bin/python3 score_explanations.py \
    --mcq /uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl \
    --results "$OUT" \
    --out "$SCORED" \
    --randomize-answers --seed 42 --resume >> "$LOGFILE" 2>&1

echo "$(date) Scoring done. Chain watcher done." >> "$LOGFILE"
