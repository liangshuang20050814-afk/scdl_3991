#!/usr/bin/env bash
set -euo pipefail

# Continue BC1 after run_bc1_overnight.sh:
# Random-20 selection seeds 2-5, three attention methods, and comparisons.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_TRAIN_GPU:-2}
EXPECTED_FOLDS=8
MANIFEST="$PROJECT_DIR/output/comparison/ST/andersson/run_manifest.tsv"
STATIC_OUTPUT="$PROJECT_DIR/output/comparison/ST/andersson/baseline_vs_random20"
ALL_OUTPUT="$PROJECT_DIR/output/comparison/ST/andersson/all_selection_methods"

cd "$PROJECT_DIR"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$MANIFEST")"
touch "$MANIFEST"

evaluated_fold_count() {
    local run_dir=$1
    find "$run_dir" -mindepth 3 -maxdepth 3 -type f \
        -path '*/eval/metrics.csv' 2>/dev/null | wc -l
}

is_complete_run() {
    local run_dir=$1
    [[ -d "$run_dir" ]] && [[ "$(evaluated_fold_count "$run_dir")" -eq "$EXPECTED_FOLDS" ]]
}

latest_complete_run() {
    local method=$1
    local run_dir
    while IFS= read -r run_dir; do
        if is_complete_run "$run_dir"; then
            printf '%s\n' "$run_dir"
            return 0
        fi
    done < <(find "$PROJECT_DIR/logs/ST/andersson/$method" \
        -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)
    return 1
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

record_run() {
    local key=$1
    local path=$2
    printf '%s\t%s\n' "$key" "$path" >> "$MANIFEST"
}

run_and_evaluate() {
    local key=$1
    local config=$2
    shift 2
    local extra_args=("$@")
    local method=${config##*/}
    local timestamp run_dir

    if run_dir=$(manifest_get "$key"); then
        echo "Reusing completed $key: $run_dir"
        LAST_RUN=$run_dir
        return 0
    fi

    echo "Training $key ($method) on physical GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" --mode cv --gpu 1 "${extra_args[@]}"

    timestamp=$(find "$PROJECT_DIR/logs/ST/andersson/$method" \
        -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)
    run_dir="$PROJECT_DIR/logs/ST/andersson/$method/$timestamp"
    [[ -d "$run_dir" ]]

    echo "Evaluating $key: $run_dir"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" --mode eval --gpu 1 --timestamp "$timestamp" \
        "${extra_args[@]}"

    if ! is_complete_run "$run_dir"; then
        echo "ERROR: $key has $(evaluated_fold_count "$run_dir")/$EXPECTED_FOLDS evaluated folds" >&2
        return 1
    fi
    record_run "$key" "$run_dir"
    LAST_RUN=$run_dir
}

echo "Waiting for bc1_overnight to finish before using GPU $GPU"
while tmux has-session -t bc1_overnight 2>/dev/null; do
    sleep 60
done

echo "Checking the four first-stage static runs"
BASELINE_RUN=$(latest_complete_run TRIPLEX) || {
    echo "ERROR: no complete 8-fold BC1 Baseline run found" >&2
    exit 1
}
RANDOM1_RUN=$(latest_complete_run TRIPLEX_random20) || {
    echo "ERROR: no complete 8-fold BC1 Random-20 seed-1 run found" >&2
    exit 1
}
SIMILARITY_RUN=$(latest_complete_run TRIPLEX_similarity20) || {
    echo "ERROR: no complete 8-fold BC1 Similarity-20 run found" >&2
    exit 1
}
DIVERSITY_RUN=$(latest_complete_run TRIPLEX_diversity20) || {
    echo "ERROR: no complete 8-fold BC1 Diversity-20 run found" >&2
    exit 1
}

record_run baseline "$BASELINE_RUN"
record_run random_seed_1 "$RANDOM1_RUN"
record_run similarity "$SIMILARITY_RUN"
record_run diversity "$DIVERSITY_RUN"

RANDOM_RUNS=("$RANDOM1_RUN")
for seed in 2 3 4 5; do
    run_and_evaluate "random_seed_$seed" ST/andersson/TRIPLEX_random20 \
        --selection_seed "$seed"
    RANDOM_RUNS+=("$LAST_RUN")
done

run_and_evaluate attention_soft ST/andersson/TRIPLEX_attention_soft
SOFT_RUN=$LAST_RUN

run_and_evaluate attention_st20 ST/andersson/TRIPLEX_attention_st20
ST_RUN=$LAST_RUN

run_and_evaluate attention_twostage20 ST/andersson/TRIPLEX_attention_twostage20 \
    --selector_ckpt_path "$SOFT_RUN"
TWOSTAGE_RUN=$LAST_RUN

echo "Generating the BC1 static-method comparison"
python src/experiment/compare_static_selections.py \
    --baseline-run "$BASELINE_RUN" \
    --random-runs "${RANDOM_RUNS[@]}" \
    --similarity-run "$SIMILARITY_RUN" \
    --diversity-run "$DIVERSITY_RUN" \
    --expected-folds "$EXPECTED_FOLDS" \
    --output-dir "$STATIC_OUTPUT"

echo "Generating the BC1 seven-method comparison"
python src/experiment/compare_all_selections.py \
    --static-fold-metrics "$STATIC_OUTPUT/fold_metrics.csv" \
    --soft-run "$SOFT_RUN" \
    --st-run "$ST_RUN" \
    --two-stage-run "$TWOSTAGE_RUN" \
    --expected-folds "$EXPECTED_FOLDS" \
    --output-dir "$ALL_OUTPUT"

echo "BC1 follow-up completed. Results: $ALL_OUTPUT"
