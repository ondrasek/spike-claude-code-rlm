#!/usr/bin/env bash
# Session start hook: dependency hygiene checks
# Runs once when a Claude Code session begins
# Non-blocking — reports issues but doesn't prevent session start

cd "${CLAUDE_PROJECT_DIR:-.}"

WARNINGS=""

# 1. Dependency hygiene (deptry) — find unused/missing/transitive deps
DEPTRY_OUTPUT=$(uv run deptry . 2>&1)
DEPTRY_EXIT=$?
if [ $DEPTRY_EXIT -ne 0 ]; then
    WARNINGS="${WARNINGS}DEPENDENCY ISSUES (deptry):\n${DEPTRY_OUTPUT}\n\n"
fi

if [ -n "$WARNINGS" ]; then
    echo -e "Session start checks found issues:\n${WARNINGS}" >&2
    echo "These are non-blocking warnings. Consider fixing them during this session." >&2
    exit 0  # Non-blocking — don't prevent session start
fi

exit 0
