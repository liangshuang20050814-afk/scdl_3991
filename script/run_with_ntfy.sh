#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <experiment-name> <command> [args ...]" >&2
    exit 2
fi

EXPERIMENT_NAME=$1
shift
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NOTIFIER="$SCRIPT_DIR/notify_ntfy.sh"
START_TIME=$(date +%s)

"$NOTIFIER" start \
    "$EXPERIMENT_NAME started on $(hostname)" \
    "TRIPLEX started" || true

"$@"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION_SECONDS=$((END_TIME - START_TIME))
DURATION_MINUTES=$((DURATION_SECONDS / 60))

if [[ $EXIT_CODE -eq 0 ]]; then
    "$NOTIFIER" success \
        "$EXPERIMENT_NAME completed in ${DURATION_MINUTES} min on $(hostname)" \
        "TRIPLEX completed" || true
else
    "$NOTIFIER" failure \
        "$EXPERIMENT_NAME failed with exit code $EXIT_CODE after ${DURATION_MINUTES} min on $(hostname)" \
        "TRIPLEX failed" || true
fi

exit "$EXIT_CODE"
