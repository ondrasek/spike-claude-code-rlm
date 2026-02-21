# Example 1: US Constitution Analysis

Analyze the full text of the US Constitution (~45KB) using a real LLM
through RLM's REPL pipeline.

## What It Does

Runs three queries against the bundled `constitution.txt`, each demonstrating
a different RLM strategy:

1. **Document structure** — Uses `re.findall()` on CONTEXT to discover all Article
   and Amendment headings (regex search, no LLM sub-calls)
2. **Bill of Rights** — Extracts a slice via `CONTEXT[start:end]` and delegates
   summarization to `llm_query()` (chunk + recursive call)
3. **Congressional powers** — Extracts Article I, Section 8 as a slice and
   uses `llm_query()` to enumerate the powers (targeted extraction)

## Usage

```bash
# Default: Ollama with qwen3-coder:32b
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
