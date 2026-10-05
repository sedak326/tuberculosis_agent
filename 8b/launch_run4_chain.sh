#!/bin/bash
# launch_run4_chain.sh — full unattended Run 4 chain: SFT-data generation
# (sharded across up to the full 8-GPU gpu4 node) -> SFT-from-Instruct QLoRA
# training -> eval -> GPT-4o scoring. No time cap anywhere in this chain;
# each SLURM submission is individually time-boxed per cluster etiquette and
# auto-resubmits/resumes until its stage is done. Mirrors Run 3's proven
# per-stage completion-marker pattern (launch_run3_followup_chain.sh etc.).
#
# gpu4 is a shared node — at launch time it was fully occupied by a
# colleague's job (ralmadamonter/boltz_newbatch, all 8 GPUs). Every stage
# here requests GPUs but SLURM queues until they're free; nothing below
# assumes exclusive access.

LOGFILE=/home/skavlak/finetuning/logs/launch_run4_chain.log
POLL_SECONDS=300
NUM_SHARDS=4
MAX_CONCURRENT_SHARDS=1   # 2 GPUs each -> capped at 2 GPUs total, per request

echo "$(date) Starting Run 4 full chain monitor (PID $$), no time cap, max ${MAX_CONCURRENT_SHARDS} concurrent shard(s)" >> "$LOGFILE"

# ---------- Stage 1: SFT data generation (up to MAX_CONCURRENT_SHARDS at a time) ----------
cd /home/skavlak/finetuning/data_prep

declare -A shard_done
for i in $(seq 0 $((NUM_SHARDS - 1))); do shard_done[$i]=0; done

gen_done=0
while [ "$gen_done" == "0" ]; do
    all_done=1
    active_shards=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_data_gen_run4$")
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        if [ "${shard_done[$i]}" == "1" ]; then
            continue
        fi
        all_done=0

        active=$(squeue -u skavlak -r -h -o "%i" 2>/dev/null | grep -cE "_${i}\$")
        if [ "$active" -gt 0 ]; then
            continue
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
            echo "$(date) Generation shard $i COMPLETE" >> "$LOGFILE"
            continue
        fi

        echo "$(date) Resubmitting generation shard $i" >> "$LOGFILE"
        sbatch --array=${i}-${i} run_generation_run4.slurm >> "$LOGFILE" 2>&1
        active_shards=$((active_shards + 1))
    done

    if [ "$all_done" == "1" ]; then
        echo "$(date) ALL $NUM_SHARDS GENERATION SHARDS COMPLETE. Concatenating." >> "$LOGFILE"
        cat /uss/skavlak/version_3/training_data_run4_shard_*.jsonl > /uss/skavlak/version_3/training_data_run4.jsonl
        n=$(wc -l < /uss/skavlak/version_3/training_data_run4.jsonl)
        echo "$(date) Concatenated $n examples into training_data_run4.jsonl." >> "$LOGFILE"
        gen_done=1
    else
        sleep "$POLL_SECONDS"
    fi
done

# ---------- Stage 2: SFT-from-Instruct QLoRA training ----------
cd /home/skavlak/finetuning/8b

train_done=0
while [ "$train_done" == "0" ]; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_sft_instruct_qlora_run4\$")
    if [ "$active" == "0" ]; then
        latest=$(ls -t /home/skavlak/finetuning/logs/sft_instruct_qlora_run4_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ] && grep -qF "SFT-from-Instruct QLoRA (Run 4) complete." "$latest"; then
            train_done=1
            echo "$(date) SFT training COMPLETE" >> "$LOGFILE"
        else
            echo "$(date) Resubmitting SFT training" >> "$LOGFILE"
            sbatch run_sft_instruct_qlora_run4.slurm >> "$LOGFILE" 2>&1
        fi
    fi
    [ "$train_done" == "0" ] && sleep "$POLL_SECONDS"
done

# ---------- Stage 3: Eval ----------
eval_done=0
while [ "$eval_done" == "0" ]; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_eval_run4_sft_instruct\$")
    if [ "$active" == "0" ]; then
        latest=$(ls -t /home/skavlak/finetuning/logs/eval_run4_sft_instruct_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ] && grep -qF "eval_run4_sft_instruct complete." "$latest"; then
            eval_done=1
            echo "$(date) Eval COMPLETE" >> "$LOGFILE"
        else
            echo "$(date) Resubmitting eval" >> "$LOGFILE"
            sbatch run_eval_run4_sft_instruct.slurm >> "$LOGFILE" 2>&1
        fi
    fi
    [ "$eval_done" == "0" ] && sleep "$POLL_SECONDS"
done

# ---------- Stage 4: GPT-4o scoring ----------
score_done=0
while [ "$score_done" == "0" ]; do
    active=$(squeue -u skavlak -h -o "%j" 2>/dev/null | grep -c "^tb_score_run4\$")
    if [ "$active" -gt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi
    n=$(wc -l < /uss/skavlak/version_3/eval_results/run4_sft_instruct_scored.jsonl 2>/dev/null || echo 0)
    if [ "$n" -ge 800 ]; then
        echo "$(date) SCORING COMPLETE ($n/800). Exiting monitor." >> "$LOGFILE"
        score_done=1
    else
        echo "$(date) Submitting scoring" >> "$LOGFILE"
        sbatch run_score_run4.slurm >> "$LOGFILE" 2>&1
        sleep "$POLL_SECONDS"
    fi
done

echo "$(date) RUN 4 CHAIN FULLY COMPLETE." >> "$LOGFILE"
