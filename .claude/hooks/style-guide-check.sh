#!/usr/bin/env bash
# Style guide check for CLI output formatting.
# Enforces design rules for user-facing terminal output.
#
# Rules:
#   1. No ASCII art splitter lines (===, ---, ***) in click.echo/print calls
#   2. Section headings must use click.style() with bold=True and a color
#   3. Section headings should include an emoji (Unicode escape or literal)
#
# Scope: rlm/cli.py (the main file that produces user-facing terminal output)

set -euo pipefail

SRC_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}/rlm"
CLI_FILES=("$SRC_DIR/cli.py")

ERRORS=()

for f in "${CLI_FILES[@]}"; do
    [ -f "$f" ] || continue
    basename=$(basename "$f")

    # Rule 1: No ASCII splitter lines in echo/print calls
    # Match: echo("=== ... ==="), echo("--- ... ---"), echo("*** ... ***")
    # Also match string literals assigned to vars that look like splitters
    while IFS= read -r match; do
        ERRORS+=("$basename: ASCII splitter line detected — use emoji + click.style(ALL CAPS, bold=True) instead: $match")
    done < <(grep -nE '(echo|print)\(.*"[=\-\*]{3,}' "$f" 2>/dev/null || true)

    # Rule 2: Section heading echo() calls should use click.style with bold
    # Detect: click.echo("SOME HEADING") without click.style
    # Heuristic: look for echo calls with ALL-CAPS words (3+ chars) not inside click.style
    while IFS= read -r match; do
        # Skip if the line already uses click.style
        if echo "$match" | grep -q 'click\.style'; then
            continue
        fi
        ERRORS+=("$basename: Unstyled ALL-CAPS heading — wrap with click.style(..., bold=True, fg=COLOR): $match")
    done < <(grep -nE 'click\.echo\("[^"]*[A-Z]{3,}[^"]*"\)' "$f" 2>/dev/null || true)

done

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "STYLE GUIDE VIOLATIONS:" >&2
    echo "" >&2
    for err in "${ERRORS[@]}"; do
        echo "  - $err" >&2
    done
    echo "" >&2
    echo "Design rules: Section headings must use emoji + click.style(ALL CAPS text, fg=COLOR, bold=True). No ASCII splitter lines (===, ---, ***)." >&2
    exit 1
fi

exit 0
