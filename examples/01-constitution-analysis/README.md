# Example 1: US Constitution Analysis

Analyze the full text of the US Constitution (~45KB) using a real LLM
through RLM's REPL pipeline.

## What It Does

Runs three queries against the bundled `constitution.txt`:

1. **Amendment summary** — Lists all 27 amendments with ratification years
   and one-sentence summaries
2. **Voting rights** — Identifies voting-rights amendments and traces the
   expansion of suffrage
3. **Checks and balances** — Maps the separation of powers across Articles I-III

## Usage

```bash
# Default: Ollama with qwen2.5-coder:32b
bash run.sh

# Specify a different model
bash run.sh ollama llama3.2

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic claude-sonnet-4-20250514
```

## Features Demonstrated

- `--context-file` for single-document input
- `--query` for specifying questions
- `--backend` selection
- `--verbose` for debug output

## Files

- `constitution.txt` — Full US Constitution with all 27 amendments (~45KB, public domain)
- `run.sh` — Shell script running 3 queries
