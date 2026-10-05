#!/bin/bash
#SBATCH --job-name=tb_eval
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_%j.err

# Usage:
#   sbatch --export=MODEL=<path>,NAME=<label>[,TOKENIZER=<path>] eval/run_eval.slurm
#
# Examples:
#   sbatch --export=MODEL=/uss/skavlak/tb_llm_v3_checkpoints/checkpoint-2175,NAME=8b_sft,TOKENIZER=meta-llama/Llama-3.1-8B-Instruct eval/run_eval.slurm
#   sbatch --export=MODEL=meta-llama/Llama-3.1-8B,NAME=8b_base eval/run_eval.slurm

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
MCQ=/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl
OUT=/uss/skavlak/tb_corpus_v3/eval_results/${NAME}.jsonl

TOKENIZER_ARG=""
if [ -n "$TOKENIZER" ]; then
    TOKENIZER_ARG="--tokenizer $TOKENIZER"
fi

echo "Model    : $MODEL"
echo "Name     : $NAME"
echo "Tokenizer: ${TOKENIZER:-$MODEL}"
echo "Output   : $OUT"

$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq              $MCQ \
    --model            $MODEL \
    $TOKENIZER_ARG \
    --out              $OUT \
    --randomize-answers \
    --seed             42

echo "Done."
