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

# --- Shared push helper: pull --rebase then push, with retry ---
do_push() {
    local max_retries=2
    local attempt=0
    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        push_output=$(git push -u origin HEAD 2>&1) && {
            echo "[auto-commit] Push successful" >&2
            debuglog "=== HOOK FINISHED — push successful (exit 0) ==="
            return 0
        }
        # Check if rejected due to remote being ahead
        if echo "$push_output" | grep -q "fetch first\|non-fast-forward"; then
            echo "[auto-commit] Remote is ahead, pulling with rebase (attempt $attempt)..." >&2
            debuglog "Pull --rebase (attempt $attempt)"
            pull_output=$(git pull --rebase origin main 2>&1) || {
                echo "[auto-commit] Pull --rebase failed: $pull_output" >&2
                debuglog "Pull --rebase failed: $pull_output"
                return 1
            }
        else
            echo "[auto-commit] Push failed: $push_output" >&2
            debuglog "Push failed (non-recoverable): $push_output"
            return 1
        fi
    done
    echo "[auto-commit] Push failed after $max_retries attempts" >&2
    debuglog "Push failed after $max_retries attempts"
    return 1
}

# Check for uncommitted changes
if [ -z "$(git status --porcelain)" ]; then
    # No uncommitted changes — but check for unpushed commits
    unpushed=$(git log @{u}..HEAD --oneline 2>/dev/null)
    if [ -n "$unpushed" ]; then
        echo "[auto-commit] No uncommitted changes, but found unpushed commits" >&2
        debuglog "Pushing unpushed commits"
        do_push || exit 0
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
do_push || exit 0

exit 0
