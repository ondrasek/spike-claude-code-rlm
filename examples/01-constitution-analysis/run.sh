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

echo "=== Example 1: US Constitution Analysis ==="
echo "Backend: $BACKEND"
echo ""

echo "--- Query 1: Amendment Summary ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "List all 27 amendments with ratification years and one-sentence summaries." \
    --verbose

echo ""
echo "--- Query 2: Voting Rights ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "Which amendments deal with voting rights? How has the right to vote expanded over time?" \
    --verbose

echo ""
echo "--- Query 3: Checks and Balances ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-file "$SCRIPT_DIR/constitution.txt" \
    --query "Identify the checks and balances described in Articles I, II, and III." \
    --verbose

echo ""
echo "=== Done ==="
