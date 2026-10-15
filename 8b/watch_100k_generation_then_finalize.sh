#!/bin/bash
# Wait for the 36k-chunk SFT generation job (4 shards) to finish, then
# concatenate it with the existing 40,252-example run4 set into a combined
# ~100k-example file for v5's SFT, and drop a marker file.

cd /home/skavlak/finetuning/8b
LOGFILE=/home/skavlak/finetuning/logs/watch_100k_data_chain.log
POLL_SECONDS=300
ADDL_DIR=/uss/skavlak/version_3/sft_100k_additional
OUT=/uss/skavlak/version_5/training_data_v5_100k.jsonl
MARKER=/uss/skavlak/version_5/.100k_data_ready

echo "$(date) Starting 100k-SFT-data finalize watcher (PID $$)" >> "$LOGFILE"

while true; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_data_gen_run4_100k\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi
    n_shards=$(ls "$ADDL_DIR"/training_data_shard_*.jsonl 2>/dev/null | wc -l)
    if [ "$n_shards" -eq 4 ]; then
        echo "$(date) All 4 shards present, no job active. Finalizing." >> "$LOGFILE"
        break
    fi
    echo "$(date) No job active but only $n_shards/4 shards found - will keep polling (resubmit manually if this persists)." >> "$LOGFILE"
    sleep "$POLL_SECONDS"
done

cat /uss/skavlak/version_3/training_data_run4.jsonl \
    "$ADDL_DIR"/training_data_shard_0.jsonl \
    "$ADDL_DIR"/training_data_shard_1.jsonl \
    "$ADDL_DIR"/training_data_shard_2.jsonl \
    "$ADDL_DIR"/training_data_shard_3.jsonl \
    > "$OUT"

n_total=$(wc -l < "$OUT")
echo "$(date) Combined SFT data written to $OUT ($n_total examples)." >> "$LOGFILE"
touch "$MARKER"
echo "$(date) 100k-data-ready marker written to $MARKER. Watcher done." >> "$LOGFILE"
