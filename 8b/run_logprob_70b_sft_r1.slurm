#!/bin/bash
#SBATCH --job-name=tb_logprob_70b_sft_r1
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/logprob_70b_sft_r1_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/logprob_70b_sft_r1_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

mkdir -p /uss/skavlak/tb_corpus_v3/eval_results

echo "=== 70b_sft_r1 (log-likelihood scoring, randomized answers, seed 42) ==="
/home/skavlak/miniconda3/envs/autorag/bin/python /home/skavlak/finetuning/data_prep/evaluate_emcqa_logprob.py \
    --model      meta-llama/Llama-3.1-70B \
    --adapter    /uss/skavlak/tb_llm_70b_v3_checkpoints/final \
    --tokenizer  meta-llama/Llama-3.1-70B-Instruct \
    --out        /uss/skavlak/tb_corpus_v3/eval_results/70b_sft_r1_logprob.jsonl \
    --randomize-answers \
    --seed       42 \
    --resume
status=$?
if [ $status -ne 0 ]; then
    echo "LOGPROB EVAL FAILED (exit $status)"
    exit $status
fi
echo "logprob_70b_sft_r1 complete."
