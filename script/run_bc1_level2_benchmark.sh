#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_LEVEL2_GPU:-2}
CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate triplex2025
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT="$PROJECT_DIR/input/ST/andersson/emb/neighbor_compact/cigar"
OUTPUT="$PROJECT_DIR/output/comparison/ST/andersson/level2_compact"
mkdir -p "$OUTPUT"

run_one() {
    local method=$1
    local config=$2
    local timestamp=$3
    local cache=$4

    echo "Validating Level 1 vs Level 2 equivalence: $method"
    CUDA_VISIBLE_DEVICES="$GPU" python \
        src/experiment/validate_level2_equivalence.py \
        --config-name "$config" \
        --timestamp "$timestamp" \
        --compact-dir "$ROOT/$cache" \
        --output "$OUTPUT/equivalence_${method}.csv" \
        --fold 0 \
        --batch-size 128

    echo "Benchmarking mask, Level 1, and Level 2: $method"
    CUDA_VISIBLE_DEVICES="$GPU" python \
        src/experiment/benchmark_triplex_level2.py \
        --config-name "$config" \
        --timestamp "$timestamp" \
        --compact-dir "$ROOT/$cache" \
        --output "$OUTPUT/benchmark_${method}.csv" \
        --fold 0 \
        --warmup 10 \
        --iterations 100 \
        --loader-batches 50 \
        --pipeline-batches 20 \
        --repeats 5
}

run_one random ST/andersson/TRIPLEX_random5 2026-10-20-17-35 random_k5_seed1
run_one similarity ST/andersson/TRIPLEX_similarity5 2026-10-20-20-17 similarity_k5_seed1
run_one diversity ST/andersson/TRIPLEX_diversity5 2026-10-20-23-05 diversity_k5_seed1

echo "BC1 Level 2 benchmarks completed"
