#!/usr/bin/env bash
set -euo pipefail

# BC1 fixed-budget screening on physical GPU 2.
# Runs Random, Similarity, Diversity, and ST Hard at K=5,10,15, then
# combines them with the existing K=20 runs and original Baseline-25.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_TRAIN_GPU:-2}
EXPECTED_FOLDS=8
MIN_FREE_GIB=${BC1_BUDGET_MIN_FREE_GIB:-30}
OUTPUT_DIR="$PROJECT_DIR/output/comparison/ST/andersson/patch_budget"
MANIFEST="$OUTPUT_DIR/run_manifest.tsv"
PRIOR_MANIFEST="$PROJECT_DIR/output/comparison/ST/andersson/run_manifest.tsv"

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
    find "$run_dir" -mindepth 3 -maxdepth 3 -type f \
        -path '*/eval/metrics.csv' 2>/dev/null | wc -l
}

is_complete_run() {
    local run_dir=$1
    [[ -d "$run_dir" ]] \
        && [[ "$(evaluated_fold_count "$run_dir")" -eq "$EXPECTED_FOLDS" ]]
}

manifest_get() {
    local manifest=$1
    local key=$2
    local path
    path=$(awk -F '\t' -v key="$key" '$1 == key {value=$2} END {print value}' \
        "$manifest")
    if [[ -n "$path" ]] && is_complete_run "$path"; then
        printf '%s\n' "$path"
        return 0
    fi
    return 1
}

record_run() {
    local key=$1
    local path=$2
    printf '%s\t%s\n' "$key" "$path" >> "$MANIFEST"
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

check_disk() {
    local free_gib
    free_gib=$(df --output=avail -BG "$PROJECT_DIR" | tail -1 | tr -dc '0-9')
    if [[ "$free_gib" -lt "$MIN_FREE_GIB" ]]; then
        echo "ERROR: only ${free_gib} GiB free; refusing to start another run" >&2
        exit 1
    fi
    echo "Disk guard passed: ${free_gib} GiB free"
}

run_and_evaluate() {
    local key=$1
    local config=$2
    local method=${config##*/}
    local timestamp run_dir

    if run_dir=$(manifest_get "$MANIFEST" "$key"); then
        echo "Reusing completed $key: $run_dir"
        LAST_RUN=$run_dir
        return
    fi

    check_disk
    wait_for_gpu
    echo "Training $key ($method) on physical GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" --mode cv --gpu 1

    timestamp=$(find "$PROJECT_DIR/logs/ST/andersson/$method" \
        -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)
    run_dir="$PROJECT_DIR/logs/ST/andersson/$method/$timestamp"
    [[ -d "$run_dir" ]]

    echo "Evaluating $key: $run_dir"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" --mode eval --gpu 1 --timestamp "$timestamp"

    if ! is_complete_run "$run_dir"; then
        echo "ERROR: $key has $(evaluated_fold_count "$run_dir")/$EXPECTED_FOLDS evaluated folds" >&2
        exit 1
    fi
    record_run "$key" "$run_dir"
    LAST_RUN=$run_dir
}

if [[ ! -r "$PRIOR_MANIFEST" ]]; then
    echo "ERROR: existing BC1 run manifest is missing: $PRIOR_MANIFEST" >&2
    exit 1
fi

BASELINE_RUN=$(manifest_get "$PRIOR_MANIFEST" baseline)
RANDOM20_RUN=$(manifest_get "$PRIOR_MANIFEST" random_seed_1)
SIMILARITY20_RUN=$(manifest_get "$PRIOR_MANIFEST" similarity)
DIVERSITY20_RUN=$(manifest_get "$PRIOR_MANIFEST" diversity)
ST20_RUN=$(manifest_get "$PRIOR_MANIFEST" attention_st20)

COMPARE_ARGS=(
    --baseline-run "$BASELINE_RUN"
    --run Random 20 "$RANDOM20_RUN"
    --run Similarity 20 "$SIMILARITY20_RUN"
    --run Diversity 20 "$DIVERSITY20_RUN"
    --run "ST Hard" 20 "$ST20_RUN"
)

for budget in 5 10 15; do
    for method in random similarity diversity attention_st; do
        key="${method}_k${budget}"
        config="ST/andersson/TRIPLEX_${method}${budget}"
        run_and_evaluate "$key" "$config"
        case "$method" in
            random) label=Random ;;
            similarity) label=Similarity ;;
            diversity) label=Diversity ;;
            attention_st) label="ST Hard" ;;
        esac
        COMPARE_ARGS+=(--run "$label" "$budget" "$LAST_RUN")
    done
done

python src/experiment/compare_patch_budgets.py \
    "${COMPARE_ARGS[@]}" \
    --expected-folds "$EXPECTED_FOLDS" \
    --output-dir "$OUTPUT_DIR"

echo "BC1 patch-budget experiment completed: $OUTPUT_DIR"
