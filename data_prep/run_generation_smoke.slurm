#!/bin/bash
#SBATCH --job-name=tb_gen_smoke
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/skavlak/finetuning/logs/gen_smoke_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/gen_smoke_%j.err

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRATCH=/scratch/skavlak/tb_gen_smoke_${SLURM_JOB_ID}
PORT=8010

echo "Scratch: $SCRATCH"
mkdir -p $SCRATCH

echo "Copying corpus to scratch..."
cp /uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl $SCRATCH/corpus.jsonl
echo "Corpus ready."

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $PORT &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

echo "Waiting for vLLM server to start..."
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

$PYTHON /home/skavlak/finetuning/data_prep/generate_training_data.py \
    --sample     36 \
    --corpus     $SCRATCH/corpus.jsonl \
    --output     /home/skavlak/finetuning/logs/smoke_test_output.jsonl \
    --base-url   "http://localhost:$PORT/v1" \
    --concurrency 4

echo "Smoke test generation complete. Shutting down vLLM server."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo "Done. Output at /home/skavlak/finetuning/logs/smoke_test_output.jsonl"
