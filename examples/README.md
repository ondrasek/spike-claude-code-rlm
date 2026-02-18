# RLM Examples

Runnable examples demonstrating RLM's capabilities on real-world scenarios.

| # | Scenario | Type | Features Demonstrated |
|---|----------|------|----------------------|
| 01 | [US Constitution Analysis](01-constitution-analysis/) | Shell script | `--context-file`, `--query`, `--backend`, `--verbose` |
| 02 | [Server Log Forensics](02-log-forensics/) | Shell script | `--max-iterations`, pattern matching, structured data |
| 03 | [Codebase Architecture Audit](03-codebase-audit/) | Shell script | `--context-dir`, `--context-glob`, `--compact`, multi-file context |
| 04 | [Literary Analysis](04-literary-analysis/) | Claude Code workflow | `/rlm:rlm` skill, recursive `llm_query()`, large text |
| 05 | [CSV Data Analysis](05-csv-data-analysis/) | Claude Code workflow | `/rlm:rlm` skill, CSV/tabular data, multi-step analysis |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (or `uvx`)

## Quick Start

Shell script examples run out of the box with the `callback` backend (no API key needed):

```bash
cd examples/01-constitution-analysis && bash run.sh
```

## Backends

Each shell script accepts an optional backend argument:

```bash
# Mock backend — no API key, good for testing the pipeline
bash run.sh callback

# Anthropic — requires ANTHROPIC_API_KEY
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic

# Ollama — requires a running Ollama server
bash run.sh ollama
```

| Backend | Requires | Command |
|---------|----------|---------|
| `callback` | Nothing | `uvx rlm` |
| `anthropic` | `ANTHROPIC_API_KEY` | `uvx rlm` |
| `ollama` | Running Ollama server | `uvx --with openai rlm` |
