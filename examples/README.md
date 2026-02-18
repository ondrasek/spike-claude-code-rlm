# RLM Examples

Runnable examples demonstrating RLM's capabilities on real-world scenarios.
Each example uses a real LLM to generate and execute analysis code in the
REPL — no mocks or fake callbacks.

| # | Scenario | Type | Features Demonstrated |
|---|----------|------|----------------------|
| 01 | [US Constitution Analysis](01-constitution-analysis/) | Shell script | `--context-file`, `--query`, `--backend`, `--verbose` |
| 02 | [Server Log Forensics](02-log-forensics/) | Shell script | `--max-iterations`, pattern matching, structured data |
| 03 | [Codebase Architecture Audit](03-codebase-audit/) | Shell script | `--context-dir`, `--context-glob`, `--compact`, multi-file context |
| 04 | [Literary Analysis](04-literary-analysis/) | Claude Code workflow | `/rlm:rlm` skill, recursive `llm_query()`, large text |
| 05 | [CSV Data Analysis](05-csv-data-analysis/) | Claude Code workflow | `/rlm:rlm` skill, CSV/tabular data, multi-step analysis |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- An LLM backend (see below)

## Quick Start

```bash
# Default: Ollama (requires a running Ollama server)
bash examples/01-constitution-analysis/run.sh

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash examples/01-constitution-analysis/run.sh anthropic
```

## Backends

All examples default to `ollama`. Pass a different backend as the first argument:

```bash
bash run.sh              # Ollama (default)
bash run.sh ollama       # Ollama (explicit)
bash run.sh anthropic    # Anthropic (requires ANTHROPIC_API_KEY)
```

| Backend | Requires |
|---------|----------|
| `ollama` (default) | Running Ollama server with a model pulled (e.g. `ollama pull llama3.2`) |
| `anthropic` | `ANTHROPIC_API_KEY` environment variable |
