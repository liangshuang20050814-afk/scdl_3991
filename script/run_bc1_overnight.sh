#!/usr/bin/env bash
set -euo pipefail

# Resumable BC1 pipeline.
# Usage:
#   bash script/run_bc1_overnight.sh prepare  # data + splits + embeddings only
#   bash script/run_bc1_overnight.sh static   # four static methods (data must exist)
#   bash script/run_bc1_overnight.sh all      # prepare, then four static methods

MODE=${1:-all}
if [[ "$MODE" != "prepare" && "$MODE" != "static" && "$MODE" != "all" ]]; then
    echo "Usage: $0 {prepare|static|all}" >&2
    exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="$PROJECT_DIR/input/ST/andersson"
EXPECTED_SLIDES=36
PREP_GPU=${BC1_PREP_GPU:-2}
TRAIN_GPU=${BC1_TRAIN_GPU:-2}
MIN_FREE_GIB=${BC1_MIN_FREE_GIB:-130}

cd "$PROJECT_DIR"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1

count_files() {
    local directory=$1
    local pattern=$2
    if [[ ! -d "$directory" ]]; then
        echo 0
        return
    fi
    find "$directory" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
}

require_count() {
    local label=$1
    local directory=$2
    local pattern=$3
    local expected=$4
    local actual
    actual=$(count_files "$directory" "$pattern")
    if [[ "$actual" -ne "$expected" ]]; then
        echo "ERROR: $label: expected $expected files, found $actual in $directory" >&2
        exit 1
    fi
    echo "Verified $label: $actual files"
}

prepare_bc1() {
    local neighbor_count free_gib
    neighbor_count=$(count_files "$DATA_DIR/patches/neighbor" '*.h5')
    if [[ "$neighbor_count" -lt "$EXPECTED_SLIDES" ]]; then
        free_gib=$(df --output=avail -BG "$PROJECT_DIR" | tail -1 | tr -dc '0-9')
        echo "Free space before BC1 neighbor generation: ${free_gib} GiB"
        if [[ "$free_gib" -lt "$MIN_FREE_GIB" ]]; then
            echo "ERROR: BC1 preparation requires at least ${MIN_FREE_GIB} GiB free while neighbor files are incomplete." >&2
            exit 1
        fi
    fi

    echo "[1/5] Downloading/preparing BC1 target and neighbor patches"
    python src/preprocess/prepare_data.py \
        --input_dir "$DATA_DIR" \
        --output_dir "$DATA_DIR" \
        --mode hest \
        --save_neighbors

    require_count "target patches" "$DATA_DIR/patches" '*.h5' "$EXPECTED_SLIDES"
    require_count "neighbor patches" "$DATA_DIR/patches/neighbor" '*.h5' "$EXPECTED_SLIDES"
    require_count "AnnData files" "$DATA_DIR/adata" '*.h5ad' "$EXPECTED_SLIDES"

    echo "[2/5] Building BC1 gene sets"
    python src/preprocess/get_geneset.py \
        --st_dir "$DATA_DIR/adata" \
        --output_dir "$DATA_DIR" \
        --n_top_hvg 50 \
        --n_top_heg 1000

    test -s "$DATA_DIR/mean_1000genes.json"

    echo "[3/5] Building 8 patient-level folds"
    if [[ ! -d "$DATA_DIR/splits" ]]; then
        python src/preprocess/split_data.py --input_dir "$DATA_DIR" --n_splits 8
    fi
    require_count "training split files" "$DATA_DIR/splits" 'train_*.csv' 8
    require_count "test split files" "$DATA_DIR/splits" 'test_*.csv' 8

    echo "[4/5] Extracting CIGAR target embeddings on physical GPU $PREP_GPU"
    CUDA_VISIBLE_DEVICES="$PREP_GPU" python src/preprocess/extract_img_features.py \
        --wsi_dataroot "$DATA_DIR/wsis" \
        --patch_dataroot "$DATA_DIR/patches" \
        --embed_dataroot "$DATA_DIR/emb/global" \
        --slide_ext .tif \
        --num_n 1 \
        --model_name cigar

    echo "[5/5] Extracting CIGAR neighbor embeddings on physical GPU $PREP_GPU"
    CUDA_VISIBLE_DEVICES="$PREP_GPU" python src/preprocess/extract_img_features.py \
        --wsi_dataroot "$DATA_DIR/wsis" \
        --patch_dataroot "$DATA_DIR/patches/neighbor" \
        --embed_dataroot "$DATA_DIR/emb/neighbor" \
        --slide_ext .tif \
        --num_n 5 \
        --model_name cigar

    require_count "global CIGAR embeddings" "$DATA_DIR/emb/global/cigar" '*.h5' "$EXPECTED_SLIDES"
    require_count "neighbor CIGAR embeddings" "$DATA_DIR/emb/neighbor/cigar" '*.h5' "$EXPECTED_SLIDES"
    echo "BC1 preparation completed."
    df -h "$PROJECT_DIR"
}

verify_training_inputs() {
    require_count "target patches" "$DATA_DIR/patches" '*.h5' "$EXPECTED_SLIDES"
    require_count "AnnData files" "$DATA_DIR/adata" '*.h5ad' "$EXPECTED_SLIDES"
    require_count "training split files" "$DATA_DIR/splits" 'train_*.csv' 8
    require_count "test split files" "$DATA_DIR/splits" 'test_*.csv' 8
    require_count "global CIGAR embeddings" "$DATA_DIR/emb/global/cigar" '*.h5' "$EXPECTED_SLIDES"
    require_count "neighbor CIGAR embeddings" "$DATA_DIR/emb/neighbor/cigar" '*.h5' "$EXPECTED_SLIDES"
    test -s "$DATA_DIR/mean_1000genes.json"
}

run_method() {
    local config=$1
    local gpu=$2
    shift 2
    local extra_args=("$@")
    local method=${config##*/}
    local timestamp

    echo "Training $method on physical GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python src/main.py \
        --config_name "$config" --mode cv --gpu 1 "${extra_args[@]}"

    timestamp=$(find "$PROJECT_DIR/logs/ST/andersson/$method" -mindepth 1 -maxdepth 1 \
        -type d -printf '%f\n' | sort | tail -1)
    if [[ -z "$timestamp" ]]; then
        echo "ERROR: no timestamp found for $method" >&2
        return 1
    fi

    echo "Evaluating $method timestamp $timestamp on physical GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python src/main.py \
        --config_name "$config" --mode eval --gpu 1 --timestamp "$timestamp" \
        "${extra_args[@]}"
}

run_static_methods() {
    verify_training_inputs

    echo "Running all static methods sequentially on physical GPU $TRAIN_GPU"
    run_method ST/andersson/TRIPLEX "$TRAIN_GPU"
    run_method ST/andersson/TRIPLEX_random20 "$TRAIN_GPU" --selection_seed 1
    run_method ST/andersson/TRIPLEX_similarity20 "$TRAIN_GPU"
    run_method ST/andersson/TRIPLEX_diversity20 "$TRAIN_GPU"

    echo "BC1 static-method training and evaluation completed."
}

case "$MODE" in
    prepare)
        prepare_bc1
        ;;
    static)
        run_static_methods
        ;;
    all)
        prepare_bc1
        run_static_methods
        ;;
esac
