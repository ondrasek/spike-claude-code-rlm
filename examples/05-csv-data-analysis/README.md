# Example 5: CSV Data Analysis (Claude Code Workflow)

Analyze CO2 emissions data using the RLM Claude Code plugin for multi-step tabular analysis.

## What It Does

1. Generates a synthetic dataset of ~500 rows of CO2 emissions data
2. Uses the `/rlm:rlm` skill for progressive analysis:
   - **Schema exploration** — Understand the dataset structure
   - **Trend analysis** — Which countries reduced or increased emissions?
   - **Per-capita comparison** — Regional per-capita emissions
   - **GDP-emissions correlation** — Relationship between wealth and emissions

## Features Demonstrated

- `/rlm:rlm` Claude Code plugin skill
- CSV/tabular data processing
- Pre-imported `json` and `collections` modules in REPL
- Multi-step analysis workflow

## Prerequisites

- Claude Code with the RLM plugin installed (see [plugin docs](../../plugin/README.md))
- `uv` on PATH

## Usage

1. Generate the dataset: `python generate_dataset.py`
2. Follow the steps in `claude-workflow.md` inside Claude Code

## Files

- `generate_dataset.py` — Deterministic CSV generator (seeded RNG, stdlib only)
- `claude-workflow.md` — Step-by-step Claude Code workflow
