# Example 3: Codebase Architecture Audit

RLM auditing its own source code — a dogfooding example using multi-file
context. Works out of the box — no API key needed.

## What It Does

Points RLM at its own `rlm/` package and produces a structured architecture
report covering:

- Module structure (files, line counts, classes, functions)
- Internal dependency graph
- Design patterns detected (Strategy, Factory, ABC, etc.)
- Public API surface
- Potential improvements

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

- `--context-dir` for directory-based context loading
- `--context-glob` for filtering file patterns
- `--compact` for the shorter system prompt
- Multi-file context (`CompositeContext`): `CONTEXT.files`, `CONTEXT.file()`

## Files

- `run.py` — Python script with smart callbacks for real architecture analysis
- `run.sh` — Shell wrapper (callback delegates to `run.py`, real backends use CLI
  with two passes: full prompt vs. compact)
