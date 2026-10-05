#!/bin/bash
#SBATCH --job-name=tb_eval_run3_cpt_sft
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_run3_cpt_sft_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_run3_cpt_sft_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

echo "=== run3_cpt_sft (SFT-from-CPT QLoRA, randomized answers, seed 42) ==="
/home/skavlak/miniconda3/envs/autorag/bin/python3 /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        /uss/skavlak/version_3/mcq_eval.jsonl \
    --model      meta-llama/Llama-3.1-8B \
    --adapter    /uss/skavlak/tb_sft_cpt_run3_qlora_checkpoints/final \
    --tokenizer  meta-llama/Llama-3.1-8B-Instruct \
    --out        /uss/skavlak/version_3/eval_results/run3_cpt_sft.jsonl \
    --randomize-answers \
    --seed       42 \
    --resume
status=$?
if [ $status -ne 0 ]; then
    echo "EVAL FAILED (exit $status)"
    exit $status
fi
echo "eval_run3_cpt_sft complete."
