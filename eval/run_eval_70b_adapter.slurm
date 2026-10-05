#!/bin/bash
#SBATCH --job-name=tb_eval_70b
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_70b_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_70b_%j.err

# Usage:
#   sbatch --export=ADAPTER=<path>,NAME=<label> eval/run_eval_70b_adapter.slurm
#
# Examples:
#   sbatch --export=ADAPTER=/uss/skavlak/tb_llm_70b_v3_checkpoints/final,NAME=70b_sft eval/run_eval_70b_adapter.slurm
#   sbatch --export=ADAPTER=/uss/skavlak/tb_llm_70b_miwv_v3_checkpoints/final,NAME=70b_sft_miwv eval/run_eval_70b_adapter.slurm

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
MCQ=/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl
OUT=/uss/skavlak/tb_corpus_v3/eval_results/${NAME}.jsonl

export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

echo "Adapter  : $ADAPTER"
echo "Name     : $NAME"
echo "Output   : $OUT"

$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        $MCQ \
    --model      meta-llama/Llama-3.1-70B \
    --adapter    $ADAPTER \
    --tokenizer  meta-llama/Llama-3.1-70B-Instruct \
    --out        $OUT \
    --batch-size 2 \
    --resume

echo "Done."
