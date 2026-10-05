#!/bin/bash
# auto_resubmit_generation_run4.sh — keep resubmitting SFT generation shards
# (0-3) until every shard reports "To process: 0" (fully done), using up to
# the full 8-GPU gpu4 node (4 concurrent 2-GPU shards). No time cap on the
# monitor itself — each shard submission is capped at 4h per cluster etiquette
# (CLAUDE.md: keep individual jobs under 6h) and auto-resumes on timeout.
# Same 24k-chunk sample as Run 3, only the seed pool changed. Once all 4
# shards are done, concatenates them into training_data_run4.jsonl.

cd /home/skavlak/finetuning/data_prep
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit_generation_run4.log
POLL_SECONDS=300
NUM_SHARDS=4
MAX_CONCURRENT_SHARDS=1   # 2 GPUs each -> capped at 2 GPUs total, per request

echo "$(date) Starting Run 4 SFT generation auto-resubmit monitor (PID $$), no time cap, max ${MAX_CONCURRENT_SHARDS} concurrent shard(s)" >> "$LOGFILE"

declare -A shard_done
for i in $(seq 0 $((NUM_SHARDS - 1))); do shard_done[$i]=0; done

while true; do
    all_done=1
    active_shards=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_data_gen_run4$")
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        if [ "${shard_done[$i]}" == "1" ]; then
            continue
        fi
        all_done=0

        active=$(squeue -u skavlak -r -h -o "%i" 2>/dev/null | grep -cE "_${i}\$")
        if [ "$active" -gt 0 ]; then
            continue  # already running or queued, leave it alone
        fi

        if [ "$active_shards" -ge "$MAX_CONCURRENT_SHARDS" ]; then
            continue   # at the concurrency cap this cycle, leave this shard for later
        fi

        latest_log=$(ls -t /home/skavlak/finetuning/logs/generation_run4_*_${i}.log 2>/dev/null | head -1)
        to_process=""
        if [ -n "$latest_log" ]; then
            to_process=$(grep "^To process" "$latest_log" | tail -1 | grep -oE '[0-9]+$')
        fi

        if [ "$to_process" == "0" ]; then
            shard_done[$i]=1
            echo "$(date) Shard $i COMPLETE" >> "$LOGFILE"
            continue
        fi

        echo "$(date) Resubmitting shard $i" >> "$LOGFILE"
        sbatch --array=${i}-${i} run_generation_run4.slurm >> "$LOGFILE" 2>&1
        active_shards=$((active_shards + 1))
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) ALL $NUM_SHARDS SHARDS COMPLETE. Concatenating." >> "$LOGFILE"
        # NOTE: generate_training_data.py's sharding branch ignores --output's
        # basename when num-shards > 1 and always writes to
        # training_data_shard_{id}.jsonl (not training_data_run4_shard_{id}.jsonl)
        # in the same directory. This bit us in Run 4 (concatenated 0 examples
        # the first time) - use the actual hardcoded filename pattern.
        cat /uss/skavlak/version_3/training_data_shard_*.jsonl > /uss/skavlak/version_3/training_data_run4.jsonl
        n=$(wc -l < /uss/skavlak/version_3/training_data_run4.jsonl)
        echo "$(date) Concatenated $n examples into training_data_run4.jsonl. Exiting monitor." >> "$LOGFILE"
        break
    fi

    sleep "$POLL_SECONDS"
done
