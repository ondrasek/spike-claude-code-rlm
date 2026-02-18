#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-ollama}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

# Build the rlm command based on backend
if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    if [ "$BACKEND" = "ollama" ]; then
        RLM_CMD="uv run --directory $REPO_ROOT --with openai rlm"
    else
        RLM_CMD="uv run --directory $REPO_ROOT rlm"
    fi
elif [ "$BACKEND" = "ollama" ]; then
    RLM_CMD="uvx --with openai rlm"
else
    RLM_CMD="uvx rlm"
fi

QUERY="Perform an architecture review of this Python package: module dependency graph, design patterns used, public API surface, and potential improvements."

echo "=== Example 3: Codebase Architecture Audit ==="
echo "Backend: $BACKEND"
echo "Target: $REPO_ROOT/rlm"
echo ""

echo "--- Pass 1: Full System Prompt ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-dir "$REPO_ROOT/rlm" \
    --context-glob "**/*.py" \
    --query "$QUERY" \
    --verbose

echo ""
echo "--- Pass 2: Compact System Prompt ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-dir "$REPO_ROOT/rlm" \
    --context-glob "**/*.py" \
    --query "$QUERY" \
    --compact \
    --verbose

echo ""
echo "=== Done ==="
