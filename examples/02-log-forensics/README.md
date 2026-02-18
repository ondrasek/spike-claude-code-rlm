# Example 2: Server Log Forensics

Analyze synthetic Apache access logs to detect security incidents using
a real LLM through RLM's REPL pipeline.

## What It Does

1. Generates ~1000 lines of realistic Apache-style access logs with injected anomalies
2. Feeds the logs to RLM for LLM-driven security analysis

### Injected Anomalies

- **Brute force**: 50 rapid 401s from a single IP against `/admin/login`,
  followed by a successful login
- **Path traversal**: `../../etc/passwd` probes with URL encoding variants
- **SQL injection**: `' OR 1=1 --`, `UNION SELECT`, `DROP TABLE` in query params
- **3 AM traffic burst**: 30 requests from an automated scanner during off-hours

## Usage

```bash
# Default: Ollama
bash run.sh

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic
```

## Features Demonstrated

- `--max-iterations` for extended analysis
- Structured data processing in the REPL
- Pattern matching via the pre-imported `re` module

## Files

- `generate_logs.py` — Deterministic log generator (seeded RNG, stdlib only)
- `run.sh` — Generates logs then runs RLM analysis
