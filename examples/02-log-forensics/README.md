# Example 2: Server Log Forensics

Analyze synthetic Apache access logs to detect security incidents.

## What It Does

1. Generates ~1000 lines of realistic Apache-style access logs with injected anomalies
2. Feeds the logs to RLM for security analysis

### Injected Anomalies

- **Brute force**: Repeated 401 responses from a single IP against `/admin/login`
- **Path traversal**: `../../etc/passwd` probes
- **SQL injection**: `' OR 1=1 --` in query parameters
- **3 AM traffic burst**: Unusual user agent during off-hours

## Features Demonstrated

- `--max-iterations` for extended analysis
- Structured data processing in the REPL
- Pattern matching via the pre-imported `re` module

## Usage

```bash
# Default: callback backend
bash run.sh

# With Anthropic
ANTHROPIC_API_KEY=sk-... bash run.sh anthropic
```

## Files

- `generate_logs.py` — Deterministic log generator (seeded RNG, stdlib only)
- `run.sh` — Generates logs then runs RLM analysis
