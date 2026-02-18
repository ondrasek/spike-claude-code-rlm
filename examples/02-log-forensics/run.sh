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

echo "=== Example 2: Server Log Forensics ==="
echo "Backend: $BACKEND"
echo ""

# Generate synthetic logs
echo "Generating synthetic access logs..."
python3 "$SCRIPT_DIR/generate_logs.py" > "$SCRIPT_DIR/access.log"
echo "Generated $(wc -l < "$SCRIPT_DIR/access.log") log lines."
echo ""

echo "--- Security Analysis ---"
$RLM_CMD \
    --backend "$BACKEND" \
    --context-file "$SCRIPT_DIR/access.log" \
    --query "Analyze these server access logs for security incidents. Identify suspicious IPs, attack patterns, timeline of events, and severity assessment." \
    --max-iterations 10 \
    --verbose

echo ""
echo "=== Done ==="
