#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-callback}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
RLM_DIR="$REPO_ROOT/rlm"

# Use local repo if available, otherwise fall back to uvx
if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    RLM_BASE="uv run --directory $REPO_ROOT rlm"
elif [ "$BACKEND" = "ollama" ]; then
    RLM_BASE="uvx --with openai rlm"
else
    RLM_BASE="uvx rlm"
fi

RLM_CMD="$RLM_BASE"

QUERY="Perform an architecture review of this Python package: module dependency graph, design patterns used, public API surface, and potential improvements."

echo "=== Example 3: Codebase Architecture Audit ==="
echo "Backend: $BACKEND"
echo "Target: $RLM_DIR"
echo ""

echo "--- Pass 1: Full System Prompt ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-dir "$RLM_DIR" \
    --context-glob "**/*.py" \
    --query "$QUERY" \
    --verbose

echo ""
echo "--- Pass 2: Compact System Prompt ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-dir "$RLM_DIR" \
    --context-glob "**/*.py" \
    --query "$QUERY" \
    --compact \
    --verbose

echo ""
echo "=== Done ==="
