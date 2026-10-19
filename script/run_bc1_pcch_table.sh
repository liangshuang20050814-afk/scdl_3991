#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate completed BC1 K=20 runs on one shared Baseline-defined HPG set,
# then add PCC(H) to the publication-style comparison table.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${BC1_TRAIN_GPU:-2}
EXPECTED_FOLDS=8
EVAL_TAG=pcch
MANIFEST="$PROJECT_DIR/output/comparison/ST/andersson/run_manifest.tsv"
BUDGET_MANIFEST="$PROJECT_DIR/output/comparison/ST/andersson/patch_budget/run_manifest.tsv"
RESULT_DIR="$PROJECT_DIR/output/comparison/ST/andersson/all_selection_methods"
BUDGET_RESULT_DIR="$PROJECT_DIR/output/comparison/ST/andersson/patch_budget"
HPG_PRED_ROOT="$PROJECT_DIR/output/pred_hpg/ST/andersson"
BASELINE_PRED="$PROJECT_DIR/output/pred/ST/andersson/TRIPLEX"
TOP_INDEX="$RESULT_DIR/baseline_top50_idx.npy"

CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate triplex2025
cd "$PROJECT_DIR"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1

if [[ ${BC1_PCCH_SKIP_WAIT:-0} != 1 ]]; then
    while tmux has-session -t bc1_patch_budget 2>/dev/null; do
        echo "Waiting for bc1_patch_budget before using physical GPU $GPU"
        sleep 60
    done
fi

manifest_get() {
    local key=$1
    awk -F '\t' -v key="$key" '$1 == key {value=$2} END {print value}' "$MANIFEST"
}

budget_manifest_get() {
    local key=$1
    awk -F '\t' -v key="$key" '$1 == key {value=$2} END {print value}' \
        "$BUDGET_MANIFEST"
}

python - "$BASELINE_PRED" "$TOP_INDEX" <<'PY'
from pathlib import Path
import sys
import numpy as np

source = Path(sys.argv[1])
output = Path(sys.argv[2])
ranks = []
for fold in range(8):
    path = source / f"fold{fold}" / "pcc_rank.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    ranks.append(np.load(path))
average_rank = np.mean(ranks, axis=0)
top = np.argsort(average_rank)[::-1][:50]
output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, top)
print(f"Saved {len(top)} Baseline-defined HPG indices to {output}")
PY

evaluation_complete() {
    local run_dir=$1
    local fold metrics
    for fold in $(seq 0 7); do
        metrics="$run_dir/fold${fold}/$EVAL_TAG/metrics.csv"
        [[ -s "$metrics" ]] || return 1
        head -n 1 "$metrics" | grep -q 'test_PearsonCorrCoef_hpg' || return 1
    done
}

evaluate_run() {
    local key=$1
    local config=$2
    local run_dir=$3
    shift 3
    local extra_args=("$@")
    local timestamp=${run_dir##*/}
    local model=${config##*/}
    local output_root="$HPG_PRED_ROOT/$key"
    local fold

    if evaluation_complete "$run_dir"; then
        echo "Reusing completed PCC(H) evaluation: $key"
        return
    fi

    for fold in $(seq 0 7); do
        mkdir -p "$output_root/$model/fold${fold}"
        cp "$TOP_INDEX" "$output_root/$model/fold${fold}/idx_top.npy"
    done

    echo "Evaluating PCC(H) for $key on physical GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python src/main.py \
        --config_name "$config" \
        --mode eval \
        --gpu 1 \
        --timestamp "$timestamp" \
        --output_dir_override "$output_root" \
        --eval_tag "$EVAL_TAG" \
        "${extra_args[@]}"

    evaluation_complete "$run_dir" || {
        echo "ERROR: incomplete PCC(H) evaluation for $key" >&2
        exit 1
    }
}

BASELINE_RUN=$(manifest_get baseline)
SIMILARITY_RUN=$(manifest_get similarity)
DIVERSITY_RUN=$(manifest_get diversity)
SOFT_RUN=$(manifest_get attention_soft)
ST_RUN=$(manifest_get attention_st20)
TWOSTAGE_RUN=$(manifest_get attention_twostage20)

evaluate_run baseline ST/andersson/TRIPLEX "$BASELINE_RUN"
for seed in 1 2 3 4 5; do
    run=$(manifest_get "random_seed_$seed")
    evaluate_run "random_seed_$seed" ST/andersson/TRIPLEX_random20 \
        "$run" --selection_seed "$seed"
done
evaluate_run similarity ST/andersson/TRIPLEX_similarity20 "$SIMILARITY_RUN"
evaluate_run diversity ST/andersson/TRIPLEX_diversity20 "$DIVERSITY_RUN"
evaluate_run attention_soft ST/andersson/TRIPLEX_attention_soft "$SOFT_RUN"
evaluate_run attention_st20 ST/andersson/TRIPLEX_attention_st20 "$ST_RUN"
evaluate_run attention_twostage20 ST/andersson/TRIPLEX_attention_twostage20 \
    "$TWOSTAGE_RUN"

COLLECT_ARGS=(
    --run Baseline baseline "$BASELINE_RUN"
    --run Similarity-20 similarity "$SIMILARITY_RUN"
    --run Diversity-20 diversity "$DIVERSITY_RUN"
    --run "Soft attention" attention_soft "$SOFT_RUN"
    --run "ST Hard-20" attention_st20 "$ST_RUN"
    --run "Two-stage Hard-20" attention_twostage20 "$TWOSTAGE_RUN"
)
for seed in 1 2 3 4 5; do
    run=$(manifest_get "random_seed_$seed")
    COLLECT_ARGS+=(--run Random-20 "seed${seed}" "$run")
done

python src/experiment/collect_pcch.py \
    "${COLLECT_ARGS[@]}" \
    --eval-tag "$EVAL_TAG" \
    --expected-folds "$EXPECTED_FOLDS" \
    --output "$RESULT_DIR/pcch_summary.csv"

python src/experiment/plot_selection20_table.py \
    --summary "$RESULT_DIR/method_summary.csv" \
    --pcc-h-summary "$RESULT_DIR/pcch_summary.csv" \
    --output-prefix "$RESULT_DIR/selection20_comparison_table"

if [[ ${BC1_PCCH_ONLY_20:-0} == 1 ]]; then
    echo "20-patch PCC(H) table completed; budget PCC(H) remains queued."
    exit 0
fi

# Extend the same Baseline-defined PCC(H) calculation to the running K study.
BUDGET_COLLECT_ARGS=(
    --run Baseline 25 baseline "$BASELINE_RUN"
    --run Random 20 random_seed_1 "$(manifest_get random_seed_1)"
    --run Similarity 20 similarity "$SIMILARITY_RUN"
    --run Diversity 20 diversity "$DIVERSITY_RUN"
    --run "ST Hard" 20 attention_st20 "$ST_RUN"
)
for budget in 5 10 15; do
    for method in random similarity diversity attention_st; do
        key="${method}_k${budget}"
        run=$(budget_manifest_get "$key")
        config="ST/andersson/TRIPLEX_${method}${budget}"
        evaluate_run "budget_${key}" "$config" "$run"
        case "$method" in
            random) label=Random ;;
            similarity) label=Similarity ;;
            diversity) label=Diversity ;;
            attention_st) label="ST Hard" ;;
        esac
        BUDGET_COLLECT_ARGS+=(--run "$label" "$budget" "$key" "$run")
    done
done

python src/experiment/collect_budget_pcch.py \
    "${BUDGET_COLLECT_ARGS[@]}" \
    --eval-tag "$EVAL_TAG" \
    --expected-folds "$EXPECTED_FOLDS" \
    --output-dir "$BUDGET_RESULT_DIR"

echo "PCC(H) table completed: $RESULT_DIR/selection20_comparison_table.png"
