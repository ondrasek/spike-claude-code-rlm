RLM Examples Plan

## Goal

Create 5 comprehensive, runnable examples demonstrating RLM on real-world scenarios. Each example uses public domain or open source data and is fully self-contained.

**Mix:** 3 standalone shell scripts + 2 Claude Code skill workflows.

---

## Directory Structure

```
examples/
├── README.md                              # Overview, prerequisites, how to run
├── 01-constitution-analysis/
│   ├── run.sh                             # Shell script (standalone)
│   ├── constitution.txt                   # US Constitution text (public domain)
│   └── README.md
├── 02-log-forensics/
│   ├── run.sh                             # Shell script (standalone)
│   ├── generate_logs.py                   # Generates realistic server logs
│   └── README.md
├── 03-codebase-audit/
│   ├── run.sh                             # Shell script (standalone)
│   └── README.md
├── 04-literary-analysis/
│   ├── claude-workflow.md                 # Claude Code workflow using /rlm:rlm skill
│   ├── fetch_text.sh                      # Helper to download a Project Gutenberg text
│   └── README.md
├── 05-csv-data-analysis/
│   ├── claude-workflow.md                 # Claude Code workflow using /rlm:rlm skill
│   ├── generate_dataset.py               # Generates sample CSV dataset
│   └── README.md
```

---

## Example 1: US Constitution Analysis (Shell Script)

**Scenario:** A legal researcher needs to analyze the full US Constitution — identify all amendments, categorize them by theme (voting rights, government structure, individual rights), find cross-references between sections, and produce a structured summary.

**What it demonstrates:**
- Basic CLI usage with `--context-file` and `--query`
- Text search and section extraction on a real document
- Running with Anthropic and Ollama backends
- Verbose mode for observing the RLM loop

**Data source:** US Constitution full text (public domain, ~45KB). Bundled directly in the example directory to avoid network dependency.

**Script outline (`run.sh`):**
1. Determine backend from `$1` arg (default: `anthropic`)
2. Run three different queries against the constitution:
   - "List all 27 amendments with their ratification years and a one-sentence summary of each"
   - "Which amendments deal with voting rights? How has the right to vote expanded over time?"
   - "Identify all checks and balances described in Articles I-III"
3. Print results with clear section headers

---

## Example 2: Server Log Forensics (Shell Script)

**Scenario:** A DevOps engineer has a large server access log and needs to investigate a potential security incident — find anomalous patterns, identify suspicious IPs, correlate timestamps, and produce an incident report.

**What it demonstrates:**
- Processing structured/semi-structured data (Apache-style access logs)
- Pattern matching and aggregation using the REPL's regex and collections modules
- Data generation for reproducible testing
- The `--max-iterations` flag for complex multi-step analysis

**Data source:** Synthetically generated Apache-style access logs (~50KB, ~1000 lines) with injected anomalies (brute force attempts, path traversal probes, unusual user agents).

**Script outline:**
1. `generate_logs.py` creates realistic logs with:
   - Normal traffic patterns (GET /index, /api/users, /static/*)
   - Injected anomalies: repeated 401s from one IP, path traversal attempts, SQL injection probes, unusual burst at 3 AM
2. `run.sh` runs RLM with query: "Analyze these server logs for security incidents. Identify suspicious IPs, attack patterns, timeline of events, and severity assessment."

---

## Example 3: Codebase Architecture Audit (Shell Script)

**Scenario:** A developer joining a new project needs to understand the architecture — module dependencies, design patterns, public API surface, and potential code smells.

**What it demonstrates:**
- `--context-dir` flag to load an entire directory of source files
- `--context-glob` to filter file types (e.g., `**/*.py`)
- Multi-file context with `FILES` dict in the REPL namespace
- The compact prompt mode (`--compact`)

**Data source:** RLM's own `rlm/` source directory (dogfooding — analyze yourself).

**Script outline:**
1. Run RLM with `--context-dir ./rlm --context-glob "**/*.py"`
2. Query: "Perform an architecture review of this Python package. Identify: (1) module dependency graph, (2) design patterns used, (3) public API surface, (4) potential improvements or code smells."
3. Show both compact and full prompt modes for comparison

---

## Example 4: Literary Analysis (Claude Code Workflow)

**Scenario:** A literature student wants to analyze a classic novel — identify themes, character arcs, narrative structure, and stylistic patterns. They use Claude Code with the RLM plugin to do this interactively.

**What it demonstrates:**
- Using the `/rlm:rlm` skill inside Claude Code
- Interactive, iterative analysis workflow
- Recursive `llm_query()` for deep passage analysis
- Working with large literary texts (~500KB+)

**Data source:** "The Adventures of Sherlock Holmes" by Arthur Conan Doyle from Project Gutenberg (public domain). A helper script downloads the text.

**Workflow document (`claude-workflow.md`):**
Provides step-by-step instructions for a Claude Code user:
1. Fetch the text: `bash examples/04-literary-analysis/fetch_text.sh`
2. Initial analysis: `/rlm:rlm sherlock_holmes.txt What are the major recurring themes across all stories in this collection?`
3. Follow-up deep dive: `/rlm:rlm sherlock_holmes.txt Analyze Holmes's deductive reasoning methodology — find 3 specific passages where he explains his method and compare them.`
4. Character analysis: `/rlm:rlm sherlock_holmes.txt Create a character relationship map for all named characters, noting which stories they appear in.`

---

## Example 5: CSV Data Analysis (Claude Code Workflow)

**Scenario:** A data analyst has a large CSV dataset of global CO2 emissions by country and needs to extract trends, find outliers, compare regions, and generate a summary report — all without loading the data into pandas or a notebook.

**What it demonstrates:**
- Processing structured tabular data (CSV) with RLM
- Using the `/rlm:rlm` skill for data exploration
- The REPL's `json` and `collections` modules for aggregation
- Multi-step analysis: explore schema -> query specifics -> synthesize

**Data source:** Synthetically generated CSV (~30KB, ~500 rows) modeled on real-world CO2 emissions data (country, year, emissions_mt, population, gdp_per_capita, energy_source_mix). `generate_dataset.py` creates this deterministically.

**Workflow document (`claude-workflow.md`):**
1. Generate data: `python examples/05-csv-data-analysis/generate_dataset.py`
2. Schema exploration: `/rlm:rlm emissions_data.csv What columns and data types are in this dataset? How many rows and what's the date range?`
3. Trend analysis: `/rlm:rlm emissions_data.csv Which countries have reduced their CO2 emissions the most over the past 20 years? Which have increased the most?`
4. Regional comparison: `/rlm:rlm emissions_data.csv Compare per-capita emissions across continents. Are there outliers?`
5. Correlation: `/rlm:rlm emissions_data.csv Is there a correlation between GDP per capita and CO2 emissions? Does it vary by region?`

---

## Implementation Notes

- **All shell scripts** will be `chmod +x` and use `#!/usr/bin/env bash` with `set -euo pipefail`
- **Backend flexibility:** Each `run.sh` accepts an optional `$1` argument to override the backend (default: `anthropic`, can pass `ollama`)
- **Self-contained data:** Examples either bundle their data or include generation scripts — no external downloads required (except example 4 which optionally fetches from Project Gutenberg)
- **Query design:** Queries are crafted to exercise different RLM capabilities (search, chunk, recursive query, aggregation, multi-file)
- **README files:** Each example has a README explaining the scenario, what RLM features it showcases, how to run it, and what to look for in the output
- **Top-level README:** Provides a table of all examples, prerequisites (uv, Python 3.11+), and quick-start commands

## Files Changed

- New directory: `examples/` with all subdirectories and files listed above
- No changes to existing source code