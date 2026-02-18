# Example 2: Server Log Forensics

Analyze synthetic Apache access logs to detect security incidents.
Works out of the box — no API key needed.

## What It Does

1. Generates ~1000 lines of realistic Apache-style access logs with injected anomalies
2. Feeds the logs to RLM for security analysis
3. Produces a detailed incident report with severity ratings

### Injected Anomalies

- **Brute force**: 50 rapid 401s from a single IP against `/admin/login`,
  followed by a successful login
- **Path traversal**: `../../etc/passwd` probes with URL encoding variants
- **SQL injection**: `' OR 1=1 --`, `UNION SELECT`, `DROP TABLE` in query params
- **3 AM traffic burst**: 30 requests from an automated scanner during off-hours

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

- `--max-iterations` for extended analysis
- Structured data processing in the REPL
- Pattern matching via the pre-imported `re` and `collections` modules

## Files

- `generate_logs.py` — Deterministic log generator (seeded RNG, stdlib only)
- `run.py` — Python script with smart callbacks for real security analysis
- `run.sh` — Shell wrapper
