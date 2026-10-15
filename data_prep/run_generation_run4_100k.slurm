#!/bin/bash
#SBATCH --job-name=tb_data_gen_run4_100k
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/skavlak/finetuning/logs/generation_run4_100k_%A_%a.log
#SBATCH --error=/home/skavlak/finetuning/logs/generation_run4_100k_%A_%a.err

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRATCH=/scratch/skavlak/tb_generation_run4_100k_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=4
PORT=$((9400 + SLURM_ARRAY_TASK_ID))

echo "Run 4 SFT scale-up: 36,000 fresh chunks (non-overlapping with the original"
echo "24k sample already in training_data_run4.jsonl), to add ~60k more examples"
echo "and bring the total SFT set to ~100k. Same seeds.json / generation pipeline"
echo "as Run 4, shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS."
echo "Port    : $PORT"
echo "Scratch : $SCRATCH"

mkdir -p $SCRATCH
echo "Copying fresh 36k-chunk corpus to scratch..."
cp /uss/skavlak/version_3/corpus_v3_train_additional_36k.jsonl $SCRATCH/corpus.jsonl
echo "Corpus ready."

OUTPUT_DIR=/uss/skavlak/version_3/sft_100k_additional
mkdir -p $OUTPUT_DIR

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

# Dedicated output dir: --output's PARENT dir is where generate_training_data.py
# actually writes training_data_shard_{id}.jsonl when sharded (ignores the
# basename given here) - keeping this in its own dir avoids colliding with/
# mutating the original run4 shard files sitting in version_3/ directly.
$PYTHON /home/skavlak/finetuning/data_prep/generate_training_data.py \
    --num-shards  $NUM_SHARDS \
    --shard-id    $SLURM_ARRAY_TASK_ID \
    --corpus      $SCRATCH/corpus.jsonl \
    --output      $OUTPUT_DIR/training_data_additional.jsonl \
    --base-url    "http://localhost:$PORT/v1" \
    --concurrency 24
status=$?

echo "Generation step exited with status $status. Shutting down vLLM server."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

if [ $status -ne 0 ]; then
    echo "GENERATION FAILED (exit $status)"
    exit $status
fi
echo "Done. Output at $OUTPUT_DIR/training_data_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
