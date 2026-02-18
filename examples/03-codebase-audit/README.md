# Example 3: Codebase Architecture Audit

RLM auditing its own source code — a dogfooding example using multi-file
context and a real LLM.

## What It Does

Points RLM at its own `rlm/` package and asks for an architecture review covering:
- Module dependency graph
- Design patterns used
- Public API surface
- Potential improvements

Runs twice: once with the full system prompt and once with `--compact` for comparison.

## Usage

```bash
# Default: Ollama
bash run.sh

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic
```

## Features Demonstrated

- `--context-dir` for directory-based context loading
- `--context-glob` for filtering file patterns
- `--compact` for the shorter system prompt
- Multi-file context (`CompositeContext`): `CONTEXT.files`, `CONTEXT.file()`

## Files

- `run.sh` — Runs two audit passes (full prompt vs. compact)
