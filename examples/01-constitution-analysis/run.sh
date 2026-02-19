#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-ollama}"
MODEL="${2:-qwen2.5-coder:32b}"
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

QUERY1="Find the major sections of this document. \
Note: Article headings use mixed case (e.g. 'Article 1', 'ARTICLE TWO'). \
Step 1: Use CONTEXT.findall(r'^ARTICLE\\s+\\S+$', re.MULTILINE | re.IGNORECASE) to find Article headings. \
Step 2: Use CONTEXT.findall(r'^Amendment [IVXLC]+', re.MULTILINE) to find Amendment headings. \
Step 3: Print both lists. \
Step 4: Call FINAL() with a numbered list of every heading you found, Articles first then Amendments."

QUERY2="What rights does the Bill of Rights protect? \
The Bill of Rights text starts around byte offset 27000 and runs about 2700 bytes. \
Step 1: Grab the chunk with chunk = CONTEXT.chunk(27000, 2800) and print it to verify. \
Step 2: Pass the chunk to llm_query('Summarize each of the 10 amendments (I-X) in one sentence each.'). \
Step 3: Call FINAL() with the result from llm_query."

QUERY3="What specific powers does Congress have? \
Section 8 of Article 1 starts around byte offset 8500 and runs about 2600 bytes. \
Step 1: Grab the chunk with chunk = CONTEXT.chunk(8500, 2700) and print it to verify. \
Step 2: Pass the chunk to llm_query('List each enumerated power of Congress in this text.'). \
Step 3: Call FINAL() with the result from llm_query."

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
