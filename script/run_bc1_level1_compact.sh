#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the four retained K=5 methods with real K-token tensors.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_TRAIN_GPU:-2}
EXPECTED_FOLDS=8
MIN_FREE_GIB=${BC1_LEVEL1_MIN_FREE_GIB:-50}
OUTPUT_DIR="$PROJECT_DIR/output/comparison/ST/andersson/level1_compact/trained"
MANIFEST="$OUTPUT_DIR/run_manifest.tsv"
TOP_INDEX="$PROJECT_DIR/output/comparison/ST/andersson/all_selection_methods/baseline_top50_idx.npy"
PRED_ROOT="$OUTPUT_DIR/eval_data"
PCCH_ROOT="$OUTPUT_DIR/pcch_data"

CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate triplex2025
cd "$PROJECT_DIR"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
touch "$MANIFEST"

evaluated_fold_count() {
    local run_dir=$1
    local eval_tag=$2
    find "$run_dir" -mindepth 3 -maxdepth 3 -type f \
        -path "*/${eval_tag}/metrics.csv" 2>/dev/null | wc -l
}

is_complete_run() {
    local run_dir=$1
    [[ -d "$run_dir" ]] \
        && [[ "$(evaluated_fold_count "$run_dir" eval)" -eq "$EXPECTED_FOLDS" ]] \
        && [[ "$(evaluated_fold_count "$run_dir" pcch)" -eq "$EXPECTED_FOLDS" ]]
}

manifest_get() {
    local key=$1
    local path
    path=$(awk -F '\t' -v key="$key" '$1 == key {value=$2} END {print value}' \
        "$MANIFEST")
    if [[ -n "$path" ]] && is_complete_run "$path"; then
        printf '%s\n' "$path"
        return 0
    fi
    return 1
}

check_disk() {
    local free_gib
    free_gib=$(df --output=avail -BG "$PROJECT_DIR" | tail -1 | tr -dc '0-9')
    if [[ "$free_gib" -lt "$MIN_FREE_GIB" ]]; then
        echo "ERROR: only ${free_gib} GiB free; stopping Level 1 training" >&2
        exit 1
    fi
    echo "Disk guard passed: ${free_gib} GiB free"
}

wait_for_gpu() {
    local active
    while true; do
        active=$(nvidia-smi -i "$GPU" --query-compute-apps=pid \
            --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
        if [[ "$active" -eq 0 ]]; then
            return
        fi
        echo "Physical GPU $GPU is busy; waiting 60 seconds"
        sleep 60
    done
}

evaluate_pcch() {
    local key=$1
    local config=$2
    local run_dir=$3
    local timestamp=${run_dir##*/}
    local model=${config##*/}
    local fold
    [[ -r "$TOP_INDEX" ]]
    for fold in $(seq 0 7); do
        mkdir -p "$PCCH_ROOT/$key/$model/fold${fold}"
        cp "$TOP_INDEX" "$PCCH_ROOT/$key/$model/fold${fold}/idx_top.npy"
    done
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" \
        --mode eval \
        --gpu 1 \
        --timestamp "$timestamp" \
        --compact_neighbor_tokens \
        --no_save_predictions \
        --output_dir_override "$PCCH_ROOT/$key" \
        --eval_tag pcch
}

run_method() {
    local key=$1
    local config=$2
    local method=${config##*/}
    local timestamp run_dir

    if run_dir=$(manifest_get "$key"); then
        echo "Reusing completed $key: $run_dir"
        return
    fi

    check_disk
    wait_for_gpu
    echo "Training compact $key on physical GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" \
        --mode cv \
        --gpu 1 \
        --compact_neighbor_tokens

    timestamp=$(find "$PROJECT_DIR/logs/ST/andersson/$method" \
        -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)
    run_dir="$PROJECT_DIR/logs/ST/andersson/$method/$timestamp"
    grep -q 'compact_neighbor_tokens: true' "$run_dir/config.yaml"

    echo "Evaluating compact $key"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" \
        --mode eval \
        --gpu 1 \
        --timestamp "$timestamp" \
        --compact_neighbor_tokens \
        --no_save_predictions \
        --output_dir_override "$PRED_ROOT/$key" \
        --eval_tag eval

    evaluate_pcch "$key" "$config" "$run_dir"
    is_complete_run "$run_dir"
    printf '%s\t%s\n' "$key" "$run_dir" >> "$MANIFEST"
}

run_method random_k5 ST/andersson/TRIPLEX_random5
run_method similarity_k5 ST/andersson/TRIPLEX_similarity5
run_method diversity_k5 ST/andersson/TRIPLEX_diversity5
run_method attention_st_k5 ST/andersson/TRIPLEX_attention_st5

echo "BC1 Level 1 compact K=5 training completed: $OUTPUT_DIR"
