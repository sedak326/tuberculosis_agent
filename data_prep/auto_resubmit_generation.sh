#!/bin/bash
# auto_resubmit_generation.sh — keep resubmitting SFT generation shards (0-15) until
# every shard reports "To process: 0" (fully done). No time cap — runs unattended
# until the whole corpus is processed. Use `kill <PID>` (see logs/auto_resubmit.log
# for the PID) to stop it early if needed; that only stops new submissions, it does
# not cancel anything already running/queued.

cd /home/skavlak/finetuning/data_prep
LOGFILE=/home/skavlak/finetuning/logs/auto_resubmit.log
POLL_SECONDS=300

echo "$(date) Starting auto-resubmit monitor (PID $$), no time cap" >> "$LOGFILE"

declare -A shard_done
for i in $(seq 0 15); do shard_done[$i]=0; done

while true; do
    all_done=1
    for i in $(seq 0 15); do
        if [ "${shard_done[$i]}" == "1" ]; then
            continue
        fi
        all_done=0

        active=$(squeue -u skavlak -r -h -o "%i" 2>/dev/null | grep -cE "_${i}\$")
        if [ "$active" -gt 0 ]; then
            continue  # already running or queued, leave it alone
        fi

        latest_log=$(ls -t /home/skavlak/finetuning/logs/generation_*_${i}.log 2>/dev/null | head -1)
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
        sbatch --array=${i}-${i} run_generation.slurm >> "$LOGFILE" 2>&1
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) ALL 16 SHARDS COMPLETE. Exiting monitor." >> "$LOGFILE"
        break
    fi

    sleep "$POLL_SECONDS"
done
