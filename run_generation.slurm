#!/bin/bash
#SBATCH --job-name=tb_data_gen
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/home/skavlak/finetuning/logs/generation_%A_%a.log
#SBATCH --error=/home/skavlak/finetuning/logs/generation_%A_%a.err

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRATCH=/scratch/skavlak/tb_generation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=16
PORT=$((8000 + SLURM_ARRAY_TASK_ID))

echo "Array task : $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "Port       : $PORT"
echo "Scratch    : $SCRATCH"

# Set up node-local scratch
mkdir -p $SCRATCH/output

# Copy corpus to node-local scratch (61MB, fast)
echo "Copying corpus to scratch..."
cp /home/skavlak/finetuning/mtubercolosis/output/corpus.jsonl $SCRATCH/corpus.jsonl
echo "Corpus ready."

# Start vLLM server (model weights load from home cache)
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $PORT &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

# Run generation for this shard, writing output to node-local scratch
$PYTHON /home/skavlak/finetuning/generate_training_data.py \
    --num-shards  $NUM_SHARDS \
    --shard-id    $SLURM_ARRAY_TASK_ID \
    --base-url    "http://localhost:$PORT/v1" \
    --concurrency 4 \
    --corpus      $SCRATCH/corpus.jsonl \
    --output      $SCRATCH/output/training_data.jsonl

# Shut down vLLM
echo "Generation complete. Shutting down vLLM server."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

# Move output back to home
OUTPUT_HOME=/home/skavlak/finetuning/mtubercolosis/output
mkdir -p $OUTPUT_HOME
mv $SCRATCH/output/training_data_shard_${SLURM_ARRAY_TASK_ID}.jsonl \
    $OUTPUT_HOME/training_data_shard_${SLURM_ARRAY_TASK_ID}.jsonl

echo "Done. Output at $OUTPUT_HOME/training_data_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
