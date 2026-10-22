#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate triplex2025
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="$PROJECT_DIR/input/ST/andersson/emb/neighbor/cigar"
ROOT="$PROJECT_DIR/input/ST/andersson/emb/neighbor_compact/cigar"
OUTPUT="$PROJECT_DIR/output/comparison/ST/andersson/level2_compact"
mkdir -p "$OUTPUT"

for method in random similarity diversity; do
    target="$ROOT/${method}_k5_seed1"
    echo "Creating and validating $method K=5 cache at $target"
    time python src/preprocess/build_compact_neighbor_cache.py \
        --input-dir "$SOURCE" \
        --output-dir "$target" \
        --method "$method" \
        --budget 5 \
        --selection-seed 1
done

du -sh "$SOURCE" "$ROOT"/* | tee "$OUTPUT/cache_sizes.txt"
echo "BC1 Level 2 compact caches completed"
