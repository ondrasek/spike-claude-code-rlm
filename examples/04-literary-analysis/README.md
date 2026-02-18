# Example 4: Literary Analysis (Claude Code Workflow)

Analyze a full-length novel using the RLM Claude Code plugin in an interactive workflow.

## What It Does

Uses the `/rlm:rlm` skill to analyze "The Adventures of Sherlock Holmes" by Arthur Conan Doyle (~500KB, public domain via Project Gutenberg):

1. **Theme analysis** — Major themes across the collection
2. **Deductive reasoning deep-dive** — Exercises recursive `llm_query()` for multi-layer analysis
3. **Character relationship mapping** — Social network of recurring characters

## Features Demonstrated

- `/rlm:rlm` Claude Code plugin skill
- Interactive multi-step workflow
- Recursive `llm_query()` calls
- Large text processing (~500KB)

## Prerequisites

- Claude Code with the RLM plugin installed (see [plugin docs](../../plugin/README.md))
- `uv` on PATH

## Usage

1. Fetch the text: `bash fetch_text.sh`
2. Follow the steps in `claude-workflow.md` inside Claude Code

## Files

- `fetch_text.sh` — Downloads the text from Project Gutenberg
- `claude-workflow.md` — Step-by-step Claude Code workflow
