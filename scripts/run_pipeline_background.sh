#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_LABEL="full"
if [[ $# -gt 0 && "$1" != --* ]]; then
    RUN_LABEL="$1"
fi

SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PIPELINE_LOG_DIR:-$REPO_DIR/run_logs}"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/pipeline_${SAFE_LABEL}_${TIMESTAMP}.log"
PID_PATH="$LOG_DIR/pipeline_${SAFE_LABEL}_${TIMESTAMP}.pid"

{
    echo "Starting pipeline at $(date -Is)"
    echo "Repository: $REPO_DIR"
    echo "Command: bash $SCRIPT_DIR/run_pipeline.sh $*"
    echo
} > "$LOG_PATH"

nohup bash "$SCRIPT_DIR/run_pipeline.sh" "$@" >> "$LOG_PATH" 2>&1 &
PID="$!"
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started pipeline PID: $PID"
echo "Log: $LOG_PATH"
echo "PID file: $PID_PATH"
