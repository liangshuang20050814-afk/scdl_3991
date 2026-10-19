#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${TRIPLEX_NTFY_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/triplex/ntfy.env}

if [[ ! -r "$CONFIG_PATH" ]]; then
    echo "ntfy configuration is not readable: $CONFIG_PATH" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

: "${NTFY_SERVER:?NTFY_SERVER is required}"
: "${NTFY_TOPIC:?NTFY_TOPIC is required}"

STATUS=${1:-info}
MESSAGE=${2:-"TRIPLEX status update"}
TITLE=${3:-"TRIPLEX experiment"}

case "$STATUS" in
    success)
        PRIORITY=3
        TAGS=white_check_mark
        ;;
    failure)
        PRIORITY=5
        TAGS=x
        ;;
    start)
        PRIORITY=3
        TAGS=hourglass_flowing_sand
        ;;
    info)
        PRIORITY=3
        TAGS=information_source
        ;;
    *)
        echo "Unknown notification status: $STATUS" >&2
        exit 2
        ;;
esac

curl \
    --fail \
    --silent \
    --show-error \
    --output /dev/null \
    --retry 3 \
    --connect-timeout 10 \
    --max-time 30 \
    -H "Title: $TITLE" \
    -H "Priority: $PRIORITY" \
    -H "Tags: $TAGS" \
    --data-binary "$MESSAGE" \
    "${NTFY_SERVER%/}/$NTFY_TOPIC"
