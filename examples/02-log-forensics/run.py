#!/usr/bin/env python3
"""Server log forensics — self-contained RLM example.

Generates synthetic Apache access logs with injected security anomalies,
then analyzes them through the RLM pipeline.

    python run.py           # No API key needed — uses smart callbacks
    python run.py anthropic # Use Anthropic API for richer analysis
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rlm.backends import CallbackBackend  # noqa: E402
from rlm.rlm import RLM  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis code executed inside the RLM REPL.
# ---------------------------------------------------------------------------

FORENSICS_CODE = """\
text = str(CONTEXT)
lines = text.strip().split("\\n")

# Parse Apache combined log format
log_pat = (r'^(\\S+) \\S+ \\S+ \\[([^\\]]+)\\] "(\\S+) (\\S+) [^"]*"'
           r' (\\d+) (\\d+) "[^"]*" "([^"]*)"$')

Counter = collections.Counter
defaultdict = collections.defaultdict
ips = Counter()
status_by_ip = defaultdict(Counter)
paths_by_ip = defaultdict(list)
ua_by_ip = defaultdict(set)
timestamps_by_ip = defaultdict(list)
parsed = 0

for line in lines:
    m = re.match(log_pat, line)
    if not m:
        continue
    parsed += 1
    ip, ts, method, path, status, size, ua = m.groups()
    ips[ip] += 1
    status_by_ip[ip][int(status)] += 1
    paths_by_ip[ip].append(path)
    ua_by_ip[ip].add(ua)
    # Extract hour from timestamp like "15/Mar/2024:14:23:00 +0000"
    hour_match = re.search(r':(\\d{2}):\\d{2}:\\d{2}', ts)
    if hour_match:
        timestamps_by_ip[ip].append(int(hour_match.group(1)))

findings = []

# 1. Brute force detection: many 401s, especially to login endpoints
for ip, statuses in status_by_ip.items():
    n401 = statuses.get(401, 0)
    if n401 >= 10:
        login_hits = sum(1 for p in paths_by_ip[ip] if "login" in p.lower())
        n200 = statuses.get(200, 0)
        severity = "CRITICAL" if n200 > 0 else "HIGH"
        detail = f"{n401} failed auth attempts"
        if login_hits:
            detail += f", {login_hits} targeting login endpoints"
        if n200 > 0:
            detail += f" — SUCCESSFUL LOGIN DETECTED ({n200} 200 responses after failures)"
        findings.append((severity, "BRUTE FORCE", ip, detail))

# 2. Path traversal detection
for ip, paths in paths_by_ip.items():
    traversal = [p for p in paths if ".." in p or "%2f" in p.lower()]
    if traversal:
        examples = traversal[:3]
        findings.append(("HIGH", "PATH TRAVERSAL", ip,
                        f"{len(traversal)} directory traversal probes: {examples}"))

# 3. SQL injection detection
sqli_keywords = ["or 1=1", "union select", "drop table", "' or",
                 "information_schema", "1=1 --"]
for ip, paths in paths_by_ip.items():
    sqli = [p for p in paths
            if any(kw in p.lower() for kw in sqli_keywords)]
    if sqli:
        examples = sqli[:3]
        findings.append(("CRITICAL", "SQL INJECTION", ip,
                        f"{len(sqli)} injection attempts: {examples}"))

# 4. Unusual time-of-day patterns (late night bursts)
for ip, hours in timestamps_by_ip.items():
    late_night = [h for h in hours if 2 <= h <= 4]
    if len(late_night) > 10:
        findings.append(("MEDIUM", "SUSPICIOUS TIMING", ip,
                        f"{len(late_night)} requests during 2-4 AM"))

# 5. Suspicious user agents
for ip, agents in ua_by_ip.items():
    for agent in agents:
        if any(kw in agent.lower() for kw in ["scanner", "python-requests",
                                                "sqlmap", "nikto", "dirbuster"]):
            findings.append(("MEDIUM", "SUSPICIOUS USER AGENT", ip,
                            f"Automated tool detected: {agent}"))
            break

# Sort by severity
severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
findings.sort(key=lambda x: severity_order.get(x[0], 99))

# Build report
report = "SECURITY INCIDENT REPORT\\n"
report += "=" * 60 + "\\n\\n"
report += f"Log entries parsed: {parsed} / {len(lines)}\\n"
report += f"Unique IPs: {len(ips)}\\n"
report += f"Incidents detected: {len(findings)}\\n\\n"

if findings:
    report += "FINDINGS\\n" + "-" * 60 + "\\n\\n"
    for i, (sev, category, ip, detail) in enumerate(findings, 1):
        report += f"  [{sev}] #{i} {category}\\n"
        report += f"    Source IP: {ip}\\n"
        report += f"    Detail: {detail}\\n\\n"

# Suspicious IP summary
suspect_ips = set()
for _, _, ip, _ in findings:
    suspect_ips.add(ip)
if suspect_ips:
    report += "SUSPICIOUS IP SUMMARY\\n" + "-" * 60 + "\\n"
    for ip in sorted(suspect_ips):
        total = ips[ip]
        statuses = dict(status_by_ip[ip])
        agents = list(ua_by_ip[ip])
        report += f"  {ip}: {total} total requests, statuses={statuses}\\n"
        report += f"    User agents: {agents[:2]}\\n\\n"

# Top IPs by volume
report += "TOP 10 IPs BY REQUEST VOLUME\\n" + "-" * 60 + "\\n"
for ip, count in ips.most_common(10):
    marker = " *** SUSPECT" if ip in suspect_ips else ""
    report += f"  {ip}: {count} requests{marker}\\n"

FINAL(report)
"""


def _make_callback(analysis_code: str):
    """Create a callback that returns real analysis code for the REPL."""

    def callback(messages: list[dict[str, str]], model: str) -> str:
        last = messages[-1]["content"] if messages else ""
        if "Output:" in last:
            return '```python\nFINAL("See analysis output above.")\n```'
        return (
            "I'll analyze the access logs for security incidents.\n\n```python\n"
            + analysis_code
            + "```\n"
        )

    return callback


def main() -> None:
    """Generate logs and run security analysis."""
    backend = sys.argv[1] if len(sys.argv) > 1 else "callback"

    print("=== Example 2: Server Log Forensics ===")
    print(f"Backend: {backend}\n")

    # Generate logs
    print("Generating synthetic access logs...")
    gen_script = SCRIPT_DIR / "generate_logs.py"
    log_file = SCRIPT_DIR / "access.log"
    subprocess.run([sys.executable, str(gen_script)], check=True, stdout=log_file.open("w"))
    line_count = sum(1 for _ in log_file.open())
    print(f"Generated {line_count} log lines.\n")

    if backend != "callback":
        rlm_cmd = ["uv", "run", "--directory", str(REPO_ROOT), "rlm"]
        query = (
            "Analyze these server access logs for security incidents. "
            "Identify suspicious IPs, attack patterns, timeline of events, "
            "and severity assessment."
        )
        subprocess.run(
            [
                *rlm_cmd,
                "--backend",
                backend,
                "--context-file",
                str(log_file),
                "--query",
                query,
                "--max-iterations",
                "10",
                "--verbose",
            ],
            check=True,
        )
        return

    context = log_file.read_text(encoding="utf-8")
    query = (
        "Analyze these server access logs for security incidents. "
        "Identify suspicious IPs, attack patterns, timeline of events, "
        "and severity assessment."
    )

    print(f"{'=' * 70}")
    print(f"Query: {query}")
    print("=" * 70)

    cb = CallbackBackend(_make_callback(FORENSICS_CODE))
    rlm = RLM(backend=cb, verbose=True, max_iterations=10)
    result = rlm.completion(context=context, query=query)

    if result.success:
        print(f"\n{'~' * 70}")
        print("ANSWER:")
        print("~" * 70)
        print(result.answer)
    else:
        print(f"ERROR: {result.error}")

    print(f"\nStats: {result.stats.iterations} iterations, {result.stats.llm_calls} LLM calls")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
