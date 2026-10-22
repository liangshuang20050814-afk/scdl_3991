#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_LEVEL2_GPU:-2}
CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate triplex2025
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

while nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader \
    | sed '/^[[:space:]]*$/d' | grep -q .; do
    echo "Physical GPU $GPU is busy; waiting 30 seconds"
    sleep 30
done

ROOT="$PROJECT_DIR/input/ST/andersson/emb/neighbor_compact/cigar"
OUTPUT="$PROJECT_DIR/output/comparison/ST/andersson/level2_compact"

CUDA_VISIBLE_DEVICES="$GPU" python \
    src/experiment/benchmark_level2_batch_scaling.py \
    --config-name ST/andersson/TRIPLEX_random5 \
    --timestamp 2026-10-20-17-35 \
    --compact-dir "$ROOT/random_k5_seed1" \
    --output "$OUTPUT/batch_scaling_random.csv" \
    --fold 0 \
    --batch-sizes 128 256 512 1024 \
    --warmup 3 \
    --iterations 20 \
    --repeats 3

python src/experiment/summarize_level2_compaction.py
echo "BC1 Level 2 scaling and summary completed"
