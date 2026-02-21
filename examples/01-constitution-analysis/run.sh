#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-ollama}"
MODEL="${2:-qwen3-coder:32b}"
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

QUERY1="Find the major sections of this document."

QUERY2="What rights does the Bill of Rights protect? Summarize each amendment."

QUERY3="What specific powers does Congress have?"

echo "=== Example 1: US Constitution Analysis ==="
echo "Backend: $BACKEND"
echo "Model: $MODEL"
echo ""

echo "--- Query 1: Document Structure ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "$QUERY1" \
    --verbose

echo ""
echo "--- Query 2: Bill of Rights ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "$QUERY2" \
    --verbose

echo ""
echo "--- Query 3: Congressional Powers ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "$QUERY3" \
    --verbose

echo ""
echo "=== Done ==="
