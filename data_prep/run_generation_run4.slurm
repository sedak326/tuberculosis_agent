#!/bin/bash
#SBATCH --job-name=tb_data_gen_run4
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/skavlak/finetuning/logs/generation_run4_%A_%a.log
#SBATCH --error=/home/skavlak/finetuning/logs/generation_run4_%A_%a.err

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRATCH=/scratch/skavlak/tb_generation_run4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=4
PORT=$((9300 + SLURM_ARRAY_TASK_ID))

echo "Run 4 SFT generation, shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS"
echo "Same fixed 24,000-chunk sample as Run 3 (pre-truncated corpus file, so"
echo "sharding splits that exact 24k set rather than each shard re-sampling"
echo "24k independently), but sharded across up to 4 concurrent 2-GPU workers"
echo "(up to the full 8-GPU gpu4 node, however many are free) instead of"
echo "Run 3's single 2-GPU job. Only the seed pool changed vs Run 3 (fixed"
echo "this session: instruction/output mismatches, grounding errors,"
echo "miscategorization, notes-field corruption)."
echo "Port    : $PORT"
echo "Scratch : $SCRATCH"

mkdir -p $SCRATCH
echo "Copying pre-truncated 24k-chunk corpus to scratch..."
cp /uss/skavlak/version_3/corpus_v3_train_shuffled_24k.jsonl $SCRATCH/corpus.jsonl
echo "Corpus ready."

OUTPUT_HOME=/uss/skavlak/version_3
mkdir -p $OUTPUT_HOME

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

# No --sample here on purpose: the corpus file itself is already the fixed
# 24k-chunk sample, and --num-shards/--shard-id round-robin-splits it into
# ~6000-chunk partitions per shard. Output goes to a shard-specific file;
# these get concatenated into training_data_run4.jsonl once all 4 finish.
$PYTHON /home/skavlak/finetuning/data_prep/generate_training_data.py \
    --num-shards  $NUM_SHARDS \
    --shard-id    $SLURM_ARRAY_TASK_ID \
    --corpus      $SCRATCH/corpus.jsonl \
    --output      $OUTPUT_HOME/training_data_run4.jsonl \
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
echo "Done. Output at $OUTPUT_HOME/training_data_run4_shard_${SLURM_ARRAY_TASK_ID}.jsonl"
