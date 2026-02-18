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
# Default: Ollama with qwen2.5-coder:32b
bash examples/01-constitution-analysis/run.sh

# Specify a different model
bash examples/01-constitution-analysis/run.sh ollama llama3.2

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash examples/01-constitution-analysis/run.sh anthropic claude-sonnet-4-20250514
```

## Backends

All examples default to `ollama`. Pass a different backend as the first argument
and an optional model as the second:

```bash
bash run.sh                              # Ollama with qwen2.5-coder:32b (default)
bash run.sh ollama llama3.2              # Ollama with a different model
bash run.sh anthropic claude-sonnet-4-20250514  # Anthropic
```

| Backend | Requires |
|---------|----------|
| `ollama` (default) | Running Ollama server with a model pulled (default: `qwen2.5-coder:32b`) |
| `anthropic` | `ANTHROPIC_API_KEY` environment variable |

## Ollama Host Configuration

The CLI reads the `OLLAMA_HOST` environment variable to locate the Ollama server.
The devcontainer sets this to `host.docker.internal:11434` automatically. Override
it if your Ollama server runs elsewhere:

```bash
OLLAMA_HOST=myserver:11434 bash run.sh
```

You can also use `--base-url` for full control:

```bash
bash run.sh  # then pass --base-url manually via RLM_CMD if needed
```
