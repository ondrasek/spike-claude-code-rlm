# Example 1: US Constitution Analysis

Analyze the full text of the US Constitution (~45KB) using RLM's text
processing pipeline. Works out of the box — no API key needed.

## What It Does

Runs three queries against the bundled `constitution.txt`:

1. **Amendment summary** — Extracts all 27 amendments with ratification years
   and first-sentence summaries
2. **Voting rights** — Identifies voting-rights amendments and traces the
   expansion of suffrage
3. **Checks and balances** — Maps the separation of powers across Articles I-III

## Usage

```bash
# No API key needed — uses smart analysis callbacks
bash run.sh

# Or run the Python script directly
uv run python run.py

# With Anthropic for richer LLM-driven analysis
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic
```

## Features Demonstrated

- `--context-file` for single-document input
- `--query` for specifying questions
- `--backend` selection (callback, anthropic, ollama)
- `--verbose` for debug output
- Real text parsing with `re` module in the REPL sandbox

## Files

- `constitution.txt` — Full US Constitution with all 27 amendments (~45KB, public domain)
- `run.py` — Python script with smart callbacks that generate real analysis code
- `run.sh` — Shell wrapper (delegates to `run.py` for callback, CLI for real backends)
