# Example 1: US Constitution Analysis

Analyze the full text of the US Constitution using RLM's basic text processing capabilities.

## What It Does

Runs three queries against the bundled `constitution.txt`:

1. **Amendment summary** — Lists all 27 amendments with ratification years and one-sentence summaries
2. **Voting rights** — Identifies voting-rights amendments and traces the expansion of suffrage
3. **Checks and balances** — Maps the separation of powers across Articles I–III

## Features Demonstrated

- `--context-file` for single-document input
- `--query` for specifying questions
- `--backend` selection (callback, anthropic, ollama)
- `--verbose` for debug output

## Usage

```bash
# Default: callback backend (no API key)
bash run.sh

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic

# With Ollama
bash run.sh ollama
```

## Files

- `constitution.txt` — Full US Constitution text (~45KB, public domain)
- `run.sh` — Shell script running 3 queries
