#!/bin/bash
#SBATCH --job-name=tb_oldeval_v8_s1
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/oldeval_v8_s1_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/oldeval_v8_s1_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/venv/bin/python3

echo "=== oldeval v8 shard 1/2 on old 800-Q mcq_eval.jsonl ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        "/uss/skavlak/version_3/mcq_eval.jsonl" \
    --model      "/uss/skavlak/tb_cpt_v8_doll_full_instruct_slerp_merged" \
    --tokenizer  "/uss/skavlak/tb_cpt_v8_doll_full_instruct_slerp_merged" \
    --adapter    "/uss/skavlak/tb_sft_v8_doll_merged_100k_qlora_checkpoints/final" \
    --out        "/uss/skavlak/version_3/eval_results/oldeval_v8_doll_merged_sft100k_shard1.jsonl" \
    --batch-size 32 \
    --randomize-answers --seed 42 --resume \
    --shard-id 1 --num-shards 2
status=$?
if [ $status -ne 0 ]; then
    echo "SHARD 1 FAILED, exit $status"
    exit $status
fi
echo "oldeval_v8_shard1 complete."
