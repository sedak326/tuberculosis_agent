#!/bin/bash
#SBATCH --job-name=tb_rag_mm_calibrate
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/skavlak/finetuning/logs/rag_mm_calibrate_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/rag_mm_calibrate_%j.err

mkdir -p /home/skavlak/finetuning/logs
source /home/skavlak/miniconda3/etc/profile.d/conda.sh
conda activate /uss/skavlak/envs/jina_rag_env
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)
cd /home/skavlak/finetuning/rag

rm -rf /uss/skavlak/tb_rag_medimind_faiss_calib
mkdir -p /uss/skavlak/tb_rag_medimind_faiss_calib
python3 -c "
import build_index_medimind as m
m.OUT_DIR = __import__('pathlib').Path('/uss/skavlak/tb_rag_medimind_faiss_calib')
m.FLUSH_EVERY_DOCS = 20
for bs in [4, 8, 16]:
    import shutil, time
    shutil.rmtree(m.OUT_DIR, ignore_errors=True)
    m.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        m.build(limit=100, batch_size=bs, shard_id=0, num_shards=1)
        print(f'batch_size={bs}: OK, {time.time()-t0:.1f}s for 100 docs')
    except Exception as e:
        print(f'batch_size={bs}: FAILED: {e}')
"
