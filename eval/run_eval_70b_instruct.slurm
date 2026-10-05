#!/bin/bash
#SBATCH --job-name=tb_eval_70b_instruct
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/uss/skavlak/tb_corpus_v3/eval_results/logs/eval_70b_instruct_%j.log
#SBATCH --error=/uss/skavlak/tb_corpus_v3/eval_results/logs/eval_70b_instruct_%j.err

mkdir -p /uss/skavlak/tb_corpus_v3/eval_results/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
MCQ=/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl
OUT=/uss/skavlak/tb_corpus_v3/eval_results/70b_instruct.jsonl

export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

echo "Model    : meta-llama/Llama-3.1-70B-Instruct"
echo "Output   : $OUT"

$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        $MCQ \
    --model      meta-llama/Llama-3.1-70B-Instruct \
    --out        $OUT \
    --batch-size 2 \
    --max-new-tokens 200 \
    --load-in-4bit \
    --resume

echo "Done."
