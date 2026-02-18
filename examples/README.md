# RLM Examples

Runnable examples demonstrating RLM's capabilities on real-world scenarios.
All examples work out of the box with no API key — they use smart callbacks
that generate real analysis code for the REPL.

| # | Scenario | Type | Features Demonstrated |
|---|----------|------|----------------------|
| 01 | [US Constitution Analysis](01-constitution-analysis/) | Script | `--context-file`, `--query`, `--backend`, `--verbose` |
| 02 | [Server Log Forensics](02-log-forensics/) | Script | `--max-iterations`, pattern matching, structured data |
| 03 | [Codebase Architecture Audit](03-codebase-audit/) | Script | `--context-dir`, `--context-glob`, `--compact`, multi-file context |
| 04 | [Literary Analysis](04-literary-analysis/) | Claude Code workflow | `/rlm:rlm` skill, recursive `llm_query()`, large text |
| 05 | [CSV Data Analysis](05-csv-data-analysis/) | Claude Code workflow | `/rlm:rlm` skill, CSV/tabular data, multi-step analysis |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

## Quick Start

```bash
# Run with smart callbacks (no API key needed, produces real analysis)
bash examples/01-constitution-analysis/run.sh

# Or run the Python script directly
uv run python examples/01-constitution-analysis/run.py
```

## Backends

Each example defaults to `callback` (smart analysis callbacks, no API key).
Pass a backend name for real LLM analysis:

```bash
# Smart callbacks — no API key, real analysis via handcrafted REPL code
bash run.sh

# Anthropic — requires ANTHROPIC_API_KEY
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic

# Ollama — requires a running Ollama server
bash run.sh ollama
```

| Backend | Requires | Notes |
|---------|----------|-------|
| `callback` | Nothing | Smart callbacks generate real analysis code |
| `anthropic` | `ANTHROPIC_API_KEY` | Full LLM-driven analysis |
| `ollama` | Running Ollama server | Full LLM-driven analysis |
