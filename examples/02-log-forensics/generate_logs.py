#!/usr/bin/env python3
"""Generate synthetic Apache-style access logs with injected security anomalies.

Produces ~1000 deterministic log lines (seeded RNG) to stdout.
Uses only stdlib — no external dependencies.

Injected anomalies:
- Brute force: repeated 401s from a single IP against /admin/login
- Path traversal: ../../etc/passwd probes
- SQL injection: ' OR 1=1 -- in query params
- 3 AM traffic burst from unusual user agent
"""

import random
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)

# --- Configuration ---

NORMAL_IPS = [f"192.168.1.{i}" for i in range(10, 60)]
ATTACKER_IP_BRUTE = "10.99.88.77"
ATTACKER_IP_TRAVERSAL = "10.99.88.78"
ATTACKER_IP_SQLI = "10.99.88.79"
BOT_IP = "203.0.113.42"

NORMAL_PATHS = [
    "/",
    "/index.html",
    "/about",
    "/contact",
    "/products",
    "/products/1",
    "/products/2",
    "/api/v1/users",
    "/api/v1/status",
    "/images/logo.png",
    "/css/style.css",
    "/js/app.js",
    "/blog",
    "/blog/post-1",
    "/blog/post-2",
    "/faq",
    "/terms",
    "/privacy",
    "/sitemap.xml",
    "/robots.txt",
]

METHODS = ["GET", "GET", "GET", "GET", "POST", "PUT", "DELETE"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
]

SUSPICIOUS_UA = "python-requests/2.28.0 (automated-scanner)"

STATUS_WEIGHTS = {200: 70, 301: 5, 304: 10, 400: 3, 403: 2, 404: 8, 500: 2}
STATUSES = []
for code, weight in STATUS_WEIGHTS.items():
    STATUSES.extend([code] * weight)

START_TIME = datetime(2024, 3, 15, 0, 0, 0)


def fmt_time(dt: datetime) -> str:
    """Format datetime as Apache log timestamp."""
    return dt.strftime("%d/%b/%Y:%H:%M:%S +0000")


def log_line(ip: str, dt: datetime, method: str, path: str, status: int, size: int, ua: str) -> str:
    """Format a single Apache combined log line."""
    return f'{ip} - - [{fmt_time(dt)}] "{method} {path} HTTP/1.1" {status} {size} "-" "{ua}"'


def generate_normal_traffic(base_time: datetime, count: int) -> list[tuple[datetime, str]]:
    """Generate normal-looking traffic."""
    lines: list[tuple[datetime, str]] = []
    for _i in range(count):
        offset = timedelta(seconds=random.randint(0, 86400))
        dt = base_time + offset
        ip = random.choice(NORMAL_IPS)
        method = random.choice(METHODS)
        path = random.choice(NORMAL_PATHS)
        status = random.choice(STATUSES)
        size = random.randint(200, 50000)
        ua = random.choice(USER_AGENTS)
        lines.append((dt, log_line(ip, dt, method, path, status, size, ua)))
    return lines


def generate_brute_force(base_time: datetime) -> list[tuple[datetime, str]]:
    """Generate brute force login attempts: 50 rapid 401s then one 200."""
    lines: list[tuple[datetime, str]] = []
    start = base_time + timedelta(hours=14, minutes=23)
    for i in range(50):
        dt = start + timedelta(seconds=i * 2)
        line = log_line(
            ATTACKER_IP_BRUTE,
            dt,
            "POST",
            "/admin/login",
            401,
            1200,
            random.choice(USER_AGENTS),
        )
        lines.append((dt, line))
    # Successful login after brute force
    dt = start + timedelta(seconds=102)
    line = log_line(
        ATTACKER_IP_BRUTE,
        dt,
        "POST",
        "/admin/login",
        200,
        3400,
        random.choice(USER_AGENTS),
    )
    lines.append((dt, line))
    return lines


def generate_path_traversal(base_time: datetime) -> list[tuple[datetime, str]]:
    """Generate path traversal probes."""
    lines: list[tuple[datetime, str]] = []
    traversal_paths = [
        "/../../etc/passwd",
        "/..%2f..%2fetc%2fpasswd",
        "/images/../../../etc/shadow",
        "/static/../../etc/hosts",
        "/download?file=../../../../etc/passwd",
        "/download?file=....//....//etc/passwd",
    ]
    start = base_time + timedelta(hours=9, minutes=45)
    for i, path in enumerate(traversal_paths):
        dt = start + timedelta(seconds=i * 15)
        status = random.choice([400, 403, 404])
        line = log_line(ATTACKER_IP_TRAVERSAL, dt, "GET", path, status, 300, SUSPICIOUS_UA)
        lines.append((dt, line))
    return lines


def generate_sql_injection(base_time: datetime) -> list[tuple[datetime, str]]:
    """Generate SQL injection attempts."""
    lines: list[tuple[datetime, str]] = []
    sqli_paths = [
        "/products?id=1' OR 1=1 --",
        "/products?id=1; DROP TABLE users --",
        "/api/v1/users?name=admin'--",
        "/search?q=' UNION SELECT username,password FROM users --",
        "/login?user=admin&pass=' OR '1'='1",
        "/api/v1/users?sort=name; SELECT * FROM information_schema.tables --",
        "/products?category=1' AND (SELECT COUNT(*) FROM users) > 0 --",
    ]
    start = base_time + timedelta(hours=16, minutes=10)
    for i, path in enumerate(sqli_paths):
        dt = start + timedelta(seconds=i * 30)
        status = random.choice([200, 400, 500])
        line = log_line(ATTACKER_IP_SQLI, dt, "GET", path, status, 800, SUSPICIOUS_UA)
        lines.append((dt, line))
    return lines


def generate_3am_burst(base_time: datetime) -> list[tuple[datetime, str]]:
    """Generate suspicious 3 AM traffic burst."""
    lines: list[tuple[datetime, str]] = []
    start = base_time + timedelta(hours=3, minutes=5)
    paths = ["/api/v1/users", "/api/v1/status", "/admin/dashboard", "/admin/export"]
    for i in range(30):
        dt = start + timedelta(seconds=i * 3)
        path = random.choice(paths)
        line = log_line(BOT_IP, dt, "GET", path, 200, random.randint(500, 5000), SUSPICIOUS_UA)
        lines.append((dt, line))
    return lines


def main() -> None:
    """Generate all log lines, sort by timestamp, and print."""
    all_lines: list[tuple[datetime, str]] = []

    all_lines.extend(generate_normal_traffic(START_TIME, 900))
    all_lines.extend(generate_brute_force(START_TIME))
    all_lines.extend(generate_path_traversal(START_TIME))
    all_lines.extend(generate_sql_injection(START_TIME))
    all_lines.extend(generate_3am_burst(START_TIME))

    # Sort by timestamp for realistic log ordering
    all_lines.sort(key=lambda x: x[0])

    for _, line in all_lines:
        print(line)


if __name__ == "__main__":
    main()
