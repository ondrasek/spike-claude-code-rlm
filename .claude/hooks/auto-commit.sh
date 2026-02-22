#!/usr/bin/env bash
# Auto-commit hook: commits and pushes changes when Claude stops
# Uses Claude CLI to generate commit messages
# Exit code 2 = blocking (Claude will respond to fix issues)
# All output goes to stderr for Claude to display

# --- TEMPORARY DEBUG LOGGING (remove when verified) ---
HOOK_LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/hook-debug.log"
debuglog() {
    echo "[auto-commit] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "$HOOK_LOG"
}
debuglog "=== HOOK STARTED (pid=$$) ==="
# --- END DEBUG LOGGING ---

# Guard against infinite loop: exit if we're already in a hook
if [ -n "$CLAUDE_HOOK_RUNNING" ]; then
    echo "[auto-commit] Skipping (already in hook)" >&2
    debuglog "Skipping (CLAUDE_HOOK_RUNNING already set)"
    exit 0
fi
export CLAUDE_HOOK_RUNNING=1

cd "$CLAUDE_PROJECT_DIR"

# --- Shared push helper: push or fail with exit 2 and actionable instructions ---
do_push() {
    local branch
    branch=$(git rev-parse --abbrev-ref HEAD)
    push_output=$(git push -u origin HEAD 2>&1) && {
        echo "[auto-commit] Push successful" >&2
        debuglog "=== HOOK FINISHED — push successful (exit 0) ==="
        return 0
    }

    debuglog "Push failed: $push_output"

    # Diagnose the failure and give Claude Code clear instructions
    echo "" >&2
    echo "=== AUTO-COMMIT HOOK: git push failed ===" >&2
    echo "" >&2
    echo "git push output:" >&2
    echo "$push_output" >&2
    echo "" >&2

    if echo "$push_output" | grep -q "fetch first\|non-fast-forward\|rejected"; then
        echo "DIAGNOSIS: The remote branch has commits not present locally" >&2
        echo "  (likely CI-generated files such as ALGORITHM.pdf)." >&2
        echo "" >&2
        echo "ACTION REQUIRED:" >&2
        echo "  1. Run: git pull --rebase origin $branch" >&2
        echo "  2. If there are merge conflicts, resolve them" >&2
        echo "  3. Run: git push origin $branch" >&2
    elif echo "$push_output" | grep -q "Permission denied\|authentication\|403\|401"; then
        echo "DIAGNOSIS: Authentication or permission error." >&2
        echo "" >&2
        echo "ACTION REQUIRED:" >&2
        echo "  This is not something you can fix. Inform the user that" >&2
        echo "  git push failed due to authentication/permission issues." >&2
    elif echo "$push_output" | grep -q "Could not resolve host\|unable to access"; then
        echo "DIAGNOSIS: Network connectivity issue." >&2
        echo "" >&2
        echo "ACTION REQUIRED:" >&2
        echo "  This is not something you can fix. Inform the user that" >&2
        echo "  git push failed due to a network error." >&2
    else
        echo "DIAGNOSIS: Unknown push failure." >&2
        echo "" >&2
        echo "ACTION REQUIRED:" >&2
        echo "  Inspect the git push output above and attempt to resolve." >&2
        echo "  If unresolvable, inform the user." >&2
    fi

    echo "" >&2
    exit 2
}

# Check for uncommitted changes
if [ -z "$(git status --porcelain)" ]; then
    # No uncommitted changes — but check for unpushed commits
    unpushed=$(git log @{u}..HEAD --oneline 2>/dev/null)
    if [ -n "$unpushed" ]; then
        echo "[auto-commit] No uncommitted changes, but found unpushed commits" >&2
        debuglog "Pushing unpushed commits"
        do_push
        exit 0
    fi
    echo "[auto-commit] No changes to commit" >&2
    debuglog "No changes to commit (exit 0)"
    exit 0
fi

echo "[auto-commit] Detected uncommitted changes" >&2

# Stage all changes first so we can get accurate diff
git add -A

# Get diff for commit message generation (staged changes)
diff_summary=$(git diff --cached --stat)
changed_files=$(git diff --cached --name-only | head -10 | tr '\n' ', ')

echo "[auto-commit] Generating commit message..." >&2

# Try to generate commit message with Claude
commit_msg=$(echo "$diff_summary" | claude -p "Generate a concise git commit message (max 72 chars first line) for these changes. Output ONLY the commit message, no quotes or explanation:" --model sonnet 2>/dev/null) || {
    # Fallback if Claude call fails
    commit_msg="WIP: ${changed_files%, }"
    echo "[auto-commit] Using fallback commit message" >&2
}

echo "[auto-commit] Committing: $commit_msg" >&2

# Commit and push - capture pre-commit hook failures
# Use --no-gpg-sign to avoid GPG timeout in automated contexts
commit_output=$(git commit --no-gpg-sign -m "$commit_msg

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" 2>&1) || {
    # Pre-commit hook failed - return exit code 2 so Claude can fix it
    echo "" >&2
    echo "[auto-commit] Pre-commit hook failed:" >&2
    echo "$commit_output" >&2
    echo "" >&2
    echo "Please fix the issues above." >&2
    exit 2
}

echo "[auto-commit] Commit successful" >&2

echo "[auto-commit] Pushing to origin..." >&2
do_push

exit 0
