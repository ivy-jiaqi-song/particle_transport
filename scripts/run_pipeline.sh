#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "Pipeline launcher failed at line $LINENO with exit code $status" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"

RUN_LABEL="full"
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
    case "$1" in
        smoke)
            RUN_LABEL="smoke"
            EXTRA_ARGS+=("--smoke")
            shift
            ;;
        full|run)
            RUN_LABEL="full"
            shift
            ;;
    esac
fi

USER_HAS_CONFIG=0
for arg in "$@"; do
    if [[ "$arg" == --config=* ]]; then
        USER_HAS_CONFIG=1
        break
    fi
done

CONFIG_ARG=()
if [[ "$USER_HAS_CONFIG" -eq 0 ]]; then
    CONFIG_PATH="${PIPELINE_CONFIG:-}"
    if [[ -z "$CONFIG_PATH" ]]; then
        if [[ -f "$REPO_DIR/configs/run_config.local.toml" ]]; then
            CONFIG_PATH="$REPO_DIR/configs/run_config.local.toml"
        else
            CONFIG_PATH="$REPO_DIR/configs/run_config.example.toml"
        fi
    fi
    CONFIG_ARG=("--config=$CONFIG_PATH")
    echo "Using config: $CONFIG_PATH"
fi

echo "Run mode: $RUN_LABEL"
echo "Julia executable: $JULIA_BIN"
echo "Repository: $REPO_DIR"
cd "$REPO_DIR"
exec "$JULIA_BIN" "$REPO_DIR/run_pipeline.jl" "${CONFIG_ARG[@]}" "${EXTRA_ARGS[@]}" "$@"
