#!/usr/bin/env bash
# Quality gate hook for Claude Code Stop event
# Fail-fast: stops at the first failing check, outputs its full stderr/stdout.
# Exit 2 feeds stderr to Claude for automatic fixing.

set -o pipefail

HOOK_LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/hook-debug.log"
debuglog() {
    echo "[quality-gate] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$HOOK_LOG"
}
debuglog "=== HOOK STARTED (pid=$$) ==="

# Per-tool diagnostic hints. Keyed by the NAME passed to run_check/run_check_nonempty.
# These tell Claude how to investigate and fix each type of failure.
declare -A TOOL_HINTS
TOOL_HINTS=(
    [pytest]="Read the failing test file and the source it tests. Run 'uv run pytest path/to/test_file.py::TestClass::test_name -x --tb=long' to see the full traceback. Fix the source code, not the test, unless the test itself is wrong."
    [coverage]="Run 'uv run pytest --cov=rlm --cov-report=term-missing' to see which lines are uncovered. Add tests for the uncovered code paths."
    [ruff check]="Run 'uv run ruff check rlm/ --output-format=full' for detailed explanations. Most issues are auto-fixable with 'uv run ruff check --fix'. Read the file at the reported line before editing."
    [ruff format]="Run 'uv run ruff format rlm/' to auto-fix all formatting issues."
    [pyright]="Read the file at the reported line number. Check type annotations, imports, and function signatures. Run 'uv run pyright rlm/path/to/file.py' to re-check a single file after fixing."
    [mypy]="Read the file at the reported line number. Fix type annotations — add missing type params, annotate untyped defs, fix incompatible assignments. Run 'uv run mypy rlm/path/to/file.py' to re-check a single file after fixing."
    [bandit]="Read the flagged code. Common fixes: use 'secrets' module instead of random for security, avoid shell=True in subprocess calls, use parameterized queries for SQL. Run 'uv run bandit -r rlm/ -ll --format custom --msg-template \"{relpath}:{line} {test_id} {msg}\"' for concise output."
    [vulture]="The reported code is detected as unused (dead code). Read the file to verify it is truly unused. If it is, delete it. If it's used dynamically (e.g. via getattr or as a public API), add it to a vulture whitelist."
    [xenon]="The reported function has cyclomatic complexity rank C or worse (CC > 10). Read the function and extract helper functions to reduce branching. Each 'if', 'elif', 'for', 'while', 'and', 'or', 'except', ternary, and comprehension-if adds +1 CC. Target: every function at rank B or better (CC <= 10)."
    [refurb]="Run 'uv run refurb --explain ERRCODE' (e.g. 'uv run refurb --explain FURB123') to understand the suggested modernization. These are usually simple one-line replacements. Read the file at the reported line, apply the suggested fix."
    [import-linter]="Check the import layering rules in pyproject.toml under [tool.importlinter]. The error shows which import violates the dependency contract. Fix by restructuring the import or moving code to the correct layer."
    [semgrep]="The finding is a code pattern that matches a known security or correctness rule. Read the matched code and the rule ID. Fix the flagged pattern — do not suppress the rule unless it is a false positive."
    [ty]="Read the file at the reported line. Fix type errors — check annotations, return types, and argument types. Run 'uv run ty check rlm/path/to/file.py' to re-check a single file."
    [interrogate]="The reported module or function is missing a docstring. Add a one-line docstring to each flagged public function/class/module. Run 'uv run interrogate rlm/ -v --fail-under 70' to see which are missing."
    [style-guide]="CLI output formatting must follow the style guide: section headings use emoji + click.style(ALL CAPS text, fg=COLOR, bold=True). No ASCII splitter lines (===, ---, ***). See .claude/hooks/style-guide-check.sh for details."
)

fail() {
    local name="$1"
    local cmd="$2"
    local output="$3"
    local hint="${TOOL_HINTS[$name]:-}"

    echo "" >&2
    echo "QUALITY GATE FAILED [$name]:" >&2
    echo "Command: $cmd" >&2
    echo "" >&2
    echo "$output" >&2
    echo "" >&2
    if [ -n "$hint" ]; then
        echo "Hint: $hint" >&2
        echo "" >&2
    fi
    echo "ACTION REQUIRED: You MUST fix the issue shown above. Do NOT stop or explain — read the failing file, edit the source code to resolve it, and the quality gate will re-run automatically." >&2
    debuglog "=== FAILED: $name ==="
    exit 2
}

# run_check NAME COMMAND...
# Runs the command, fails fast if exit code is non-zero.
run_check() {
    local name="$1"; shift
    local cmd="$*"
    debuglog "Running $name..."
    OUTPUT=$("$@" 2>&1) || fail "$name" "$cmd" "$OUTPUT"
}

# run_check_nonempty NAME COMMAND...
# Runs the command, fails fast if exit code is non-zero AND there is output.
# Used for tools like vulture/refurb/semgrep where exit 0 with no output = clean.
run_check_nonempty() {
    local name="$1"; shift
    local cmd="$*"
    debuglog "Running $name..."
    OUTPUT=$("$@" 2>&1)
    [ -n "$OUTPUT" ] && fail "$name" "$cmd" "$OUTPUT"
}

# Checks are ordered by speed and likelihood of failure: fast/common first.

# Skip pytest/coverage if no test files exist
TEST_FILES=$(find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | grep -v ".venv" | head -1)
if [ -n "$TEST_FILES" ]; then
    run_check        "pytest"         uv run pytest -x --tb=short
    run_check        "coverage"       uv run pytest --cov=rlm --cov-report=term --cov-fail-under=80 -q
else
    debuglog "Skipping pytest/coverage (no test files found)"
fi
run_check        "ruff check"     uv run ruff check rlm/
run_check        "ruff format"    uv run ruff format --check rlm/
run_check        "pyright"        uv run pyright rlm/
run_check        "mypy"           uv run mypy rlm/
run_check        "bandit"         uv run bandit -r rlm/ -q -ll
run_check_nonempty "vulture"        uv run vulture rlm/ --min-confidence 80
run_check        "xenon"          uv run xenon --max-absolute B --max-modules A --max-average A rlm/
run_check_nonempty "refurb"         uv run refurb rlm/ --python-version 3.13
run_check        "import-linter"  uv run lint-imports
run_check_nonempty "semgrep"        uv run semgrep scan --config p/python --error --quiet rlm/
run_check        "ty"             uv run ty check rlm/
run_check        "interrogate"    uv run interrogate rlm/ -v --fail-under 70
run_check        "style-guide"   "${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/style-guide-check.sh"

debuglog "=== ALL 15 CHECKS PASSED ==="
exit 0
