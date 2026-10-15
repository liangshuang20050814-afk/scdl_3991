#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${1:-ST/andersson/TRIPLEX_random20}
NUM_GPUS=${2:-1}
shift $(( $# >= 2 ? 2 : $# ))

if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4 5)
fi

for seed in "${SEEDS[@]}"; do
    echo "Training ${CONFIG_NAME} with selection seed ${seed}"
    python src/main.py \
        --config_name "$CONFIG_NAME" \
        --mode cv \
        --gpu "$NUM_GPUS" \
        --selection_seed "$seed"
done
