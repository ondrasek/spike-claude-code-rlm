#!/usr/bin/env bash
# Claude Code status line script (bash port of PowerShell original)
# Reads JSON session data from stdin, outputs up to 4 lines:
#   Line 1 (always): Model (branch) | tokens used/total | % used | % remain | thinking
#   Line 2 (if authenticated): 5h usage bar | 7d usage bar | extra usage bar
#   Line 3 (if authenticated): Reset times for each usage bucket
#   Line 4 (if hooks configured): Configured hooks overview

# --- 1. Constants ---

# oh-my-posh inspired true-color palette
C_BLUE=$'\033[38;2;0;153;255m'
C_ORANGE=$'\033[38;2;255;176;85m'
C_GREEN=$'\033[38;2;0;160;0m'
C_CYAN=$'\033[38;2;46;149;153m'
C_RED=$'\033[38;2;255;85;85m'
C_YELLOW=$'\033[38;2;230;200;0m'
C_DIM=$'\033[38;2;80;80;80m'
C_WHITE=$'\033[38;2;220;220;220m'
C_RESET=$'\033[0m'

USAGE_CACHE="/tmp/claude-statusline-usage.json"
USAGE_CACHE_TTL=60
GIT_CACHE_TTL=5

# --- 2. Fallback trap ---

set -eE  # errexit + inherit ERR trap into subshells and command substitutions
trap 'printf "Claude"; exit 0' ERR

# --- 3. Guard: require jq ---

command -v jq >/dev/null 2>&1 || { printf 'Claude'; exit 0; }

# --- 4. Functions ---

# Portable date parsing: converts ISO 8601 timestamp to epoch seconds.
# GNU date uses -d, macOS/BSD date uses -jf.
date_to_epoch() {
    local ts=$1
    date -d "$ts" +%s 2>/dev/null && return
    # macOS/BSD: strip fractional seconds and timezone colon for %z parsing
    local clean=${ts%%.*}  # drop .NNN fractional part
    clean=${clean%Z}       # drop trailing Z if present
    date -jf "%Y-%m-%dT%H:%M:%S" "$clean" +%s 2>/dev/null && return
    return 1
}

# Portable date formatting: formats ISO 8601 timestamp with a strftime format.
date_fmt() {
    local ts=$1 fmt=$2
    date -d "$ts" "+$fmt" 2>/dev/null && return
    local clean=${ts%%.*}
    clean=${clean%Z}
    date -jf "%Y-%m-%dT%H:%M:%S" "$clean" "+$fmt" 2>/dev/null && return
    return 1
}

# Portable date arithmetic: outputs formatted date for an expression.
# GNU: date -d "$expr" "+$fmt", macOS: date -v modifier.
date_calc() {
    local expr=$1 fmt=$2
    date -d "$expr" "+$fmt" 2>/dev/null && return
    # Fallback for macOS: only supports "+1 month from 1st" used below
    if [[ "$expr" == *"+1 month"* ]]; then
        date -v1d -v+1m "+$fmt" 2>/dev/null && return
    fi
    return 1
}

# Portable file mtime: returns epoch seconds of file's last modification.
# GNU stat uses -c %Y, macOS/BSD stat uses -f %m.
file_mtime() {
    stat -c %Y "$1" 2>/dev/null && return
    stat -f %m "$1" 2>/dev/null && return
    echo 0
}

format_tokens() {
    local n=$1
    # Guard: only process numeric values to prevent awk injection
    [[ "$n" =~ ^[0-9]+$ ]] || { printf '0'; return; }
    if [ "$n" -ge 1000000 ] 2>/dev/null; then
        awk -v n="$n" 'BEGIN {v=n/1000000; if (v==int(v)) printf "%dm",v; else printf "%.1fm",v}'
    elif [ "$n" -ge 1000 ] 2>/dev/null; then
        awk -v n="$n" 'BEGIN {printf "%dk", n/1000}'
    else
        printf '%d' "$n"
    fi
}

format_number() {
    # Add comma separators locale-independently: 155000 -> 155,000
    printf '%d' "${1:-0}" | sed ':a;s/\([0-9]\)\([0-9]\{3\}\)\($\|,\)/\1,\2\3/;ta'
}

color_for_pct() {
    local pct=$1
    if [ "$pct" -ge 90 ] 2>/dev/null; then printf '%s' "$C_RED"
    elif [ "$pct" -ge 70 ] 2>/dev/null; then printf '%s' "$C_YELLOW"
    elif [ "$pct" -ge 50 ] 2>/dev/null; then printf '%s' "$C_ORANGE"
    else printf '%s' "$C_GREEN"
    fi
}

build_bar() {
    local pct=${1:-0} width=${2:-10}
    [ "$pct" -lt 0 ] 2>/dev/null && pct=0
    [ "$pct" -gt 100 ] 2>/dev/null && pct=100
    local filled=$((pct * width / 100))
    [ "$filled" -gt "$width" ] && filled=$width
    local empty=$((width - filled))
    local bar_color
    bar_color=$(color_for_pct "$pct")
    local bar=""
    if [ "$filled" -gt 0 ]; then
        bar="${bar_color}$(printf '%0.s●' $(seq 1 "$filled"))"
    fi
    if [ "$empty" -gt 0 ]; then
        bar="${bar}${C_DIM}$(printf '%0.s○' $(seq 1 "$empty"))"
    fi
    printf '%s%s' "$bar" "$C_RESET"
}

format_reset_time_relative() {
    local timestamp=$1
    [ -z "$timestamp" ] && return
    local reset_epoch now_epoch diff
    reset_epoch=$(date_to_epoch "$timestamp") || return
    now_epoch=$(date +%s)
    diff=$((reset_epoch - now_epoch))
    if [ "$diff" -gt 0 ]; then
        local hours=$((diff / 3600))
        local mins=$(((diff % 3600) / 60))
        if [ "$hours" -gt 0 ]; then
            printf '%dh%02dm' "$hours" "$mins"
        else
            printf '%dm' "$mins"
        fi
    else
        printf 'resetting...'
    fi
}

format_reset_time_absolute() {
    local timestamp=$1
    [ -z "$timestamp" ] && return
    date_fmt "$timestamp" "%b %-d, %-I:%M%P" 2>/dev/null | tr '[:upper:]' '[:lower:]'
}

# --- 5. Parse stdin JSON (single jq call) ---

input=$(cat)
[ -z "$input" ] && { printf 'Claude'; exit 0; }

IFS=$'\t' read -r DISPLAY_NAME PROJECT_DIR CTX_SIZE CTX_PCT INPUT_TOK CACHE_CREATE CACHE_READ <<< \
    "$(printf '%s' "$input" | jq -r '[
        .model.display_name // "?",
        .workspace.project_dir // ".",
        (.context_window.context_window_size // 200000),
        (.context_window.used_percentage // 0 | floor),
        (.context_window.current_usage.input_tokens // 0),
        (.context_window.current_usage.cache_creation_input_tokens // 0),
        (.context_window.current_usage.cache_read_input_tokens // 0)
    ] | @tsv' 2>/dev/null)"

DISPLAY_NAME="${DISPLAY_NAME:-?}"
CTX_SIZE="${CTX_SIZE:-200000}"
CTX_PCT="${CTX_PCT%%.*}"
CTX_PCT="${CTX_PCT:-0}"
[ "$CTX_PCT" -eq "$CTX_PCT" ] 2>/dev/null || CTX_PCT=0
INPUT_TOK="${INPUT_TOK:-0}"
CACHE_CREATE="${CACHE_CREATE:-0}"
CACHE_READ="${CACHE_READ:-0}"

# --- 6. Git branch (cached) ---

GIT_BRANCH=""
GIT_WORKTREE=""
if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Per-directory cache key: /workspaces/foo → workspaces-foo
    dir_key=$(printf '%s' "$PROJECT_DIR" | tr '/' '-' | sed 's/^-//')
    GIT_CACHE="/tmp/claude-statusline-git-${dir_key:-default}.txt"
    now=$(date +%s)
    cache_age=999
    if [ -f "$GIT_CACHE" ]; then
        cache_mtime=$(file_mtime "$GIT_CACHE")
        cache_age=$((now - cache_mtime))
    fi
    if [ "$cache_age" -ge "$GIT_CACHE_TTL" ]; then
        branch=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null)
        # Write branch (or empty) to cache so detached HEAD clears stale values
        if tmp=$(mktemp "${GIT_CACHE}.XXXXXX"); then
            printf '%s' "$branch" > "$tmp" && mv "$tmp" "$GIT_CACHE" 2>/dev/null || rm -f "$tmp"
        fi
        GIT_BRANCH="$branch"
    else
        GIT_BRANCH=$(cat "$GIT_CACHE" 2>/dev/null)
    fi

    # Detect worktree: git-dir differs from git-common-dir in worktrees
    git_dir=$(git -C "$PROJECT_DIR" rev-parse --git-dir 2>/dev/null)
    git_common=$(git -C "$PROJECT_DIR" rev-parse --git-common-dir 2>/dev/null)
    if [ -n "$git_dir" ] && [ -n "$git_common" ] && [ "$git_dir" != "$git_common" ]; then
        toplevel=$(git -C "$PROJECT_DIR" rev-parse --show-toplevel 2>/dev/null)
        GIT_WORKTREE="${toplevel##*/}"
    fi
fi

# --- 7. Context window calculations ---

CURRENT_TOKENS=$((INPUT_TOK + CACHE_CREATE + CACHE_READ))
REMAIN_PCT=$((100 - CTX_PCT))
REMAIN_TOKENS=$((CTX_SIZE - CURRENT_TOKENS))
[ "$REMAIN_TOKENS" -lt 0 ] 2>/dev/null && REMAIN_TOKENS=0

USED_FMT=$(format_tokens "$CURRENT_TOKENS")
TOTAL_FMT=$(format_tokens "$CTX_SIZE")
USED_COMMA=$(format_number "$CURRENT_TOKENS")
REMAIN_COMMA=$(format_number "$REMAIN_TOKENS")

# --- 8. Thinking status (respects settings precedence) ---

THINKING="off"
THINKING_COLOR="$C_DIM"
# Check settings files in precedence order: project-local > project-shared > user-global
for settings_file in \
    "$PROJECT_DIR/.claude/settings.local.json" \
    "$PROJECT_DIR/.claude/settings.json" \
    "$HOME/.claude/settings.json"; do
    [ -f "$settings_file" ] || continue
    thinking_val=$(jq -r '.alwaysThinkingEnabled // empty' "$settings_file" 2>/dev/null)
    if [ "$thinking_val" = "true" ]; then
        THINKING="on"
        THINKING_COLOR="$C_ORANGE"
        break
    elif [ "$thinking_val" = "false" ]; then
        break  # Explicit disable at higher-priority level stops search
    fi
done
if [ -n "$MAX_THINKING_TOKENS" ] && [ "$MAX_THINKING_TOKENS" != "0" ]; then
    THINKING="on"
    THINKING_COLOR="$C_ORANGE"
fi

# --- 9. Usage API fetch (cached) ---

HAVE_USAGE=false

fetch_usage() {
    local creds="$HOME/.claude/.credentials.json"
    [ -f "$creds" ] || return 1

    local token
    token=$(jq -r '.claudeAiOauth.accessToken // empty' "$creds" 2>/dev/null)
    [ -z "$token" ] && return 1

    local response http_code body header_file curl_status
    header_file=$(mktemp) || return 1
    chmod 600 "$header_file"
    printf 'Authorization: Bearer %s' "$token" > "$header_file"

    response=$(curl -s -m 5 -w "\n%{http_code}" \
        -H @"$header_file" \
        -H "Accept: application/json" \
        -H "Content-Type: application/json" \
        -H "User-Agent: claude-code/2.1.34" \
        -H "anthropic-beta: oauth-2025-04-20" \
        "https://api.anthropic.com/api/oauth/usage" 2>/dev/null)
    curl_status=$?
    rm -f "$header_file"
    [ $curl_status -eq 0 ] || return 1

    http_code=$(printf '%s' "$response" | tail -n1)
    body=$(printf '%s' "$response" | sed '$d')

    [ "$http_code" = "200" ] || return 1

    local tmp
    tmp=$(mktemp "${USAGE_CACHE}.XXXXXX") || return 1
    chmod 600 "$tmp"
    printf '%s' "$body" > "$tmp" && mv "$tmp" "$USAGE_CACHE" 2>/dev/null || rm -f "$tmp"
}

now=${now:-$(date +%s)}
cache_age=999
if [ -f "$USAGE_CACHE" ]; then
    cache_mtime=$(file_mtime "$USAGE_CACHE")
    cache_age=$((now - cache_mtime))
fi

if [ "$cache_age" -ge "$USAGE_CACHE_TTL" ]; then
    fetch_usage 2>/dev/null || true
fi

# --- 10. Parse usage cache ---

FIVE_HOUR_PCT=0; FIVE_HOUR_RESET=""
SEVEN_DAY_PCT=0; SEVEN_DAY_RESET=""
EXTRA_ENABLED=false; EXTRA_PCT=0; EXTRA_USED="0"; EXTRA_LIMIT="0"

if [ -f "$USAGE_CACHE" ] && [ -f "$HOME/.claude/.credentials.json" ]; then
    # Credits-to-dollars: API returns values in credits (20 credits = $1 USD)
    parsed=$(jq -r '[
        ((.five_hour.utilization // 0) | round),
        (.five_hour.resets_at // ""),
        ((.seven_day.utilization // 0) | round),
        (.seven_day.resets_at // ""),
        (.extra_usage.is_enabled // false),
        (.extra_usage | if .utilization then (.utilization | round)
            elif (.monthly_limit // 0) > 0 then ((.used_credits // 0) / .monthly_limit * 100 | round)
            else 0 end),
        ((.extra_usage.used_credits // 0) / 20 * 100 | round | . / 100),
        ((.extra_usage.monthly_limit // 0) / 20 * 100 | round | . / 100)
    ] | @tsv' "$USAGE_CACHE" 2>/dev/null) || parsed=""
    if [ -n "$parsed" ]; then
        IFS=$'\t' read -r FIVE_HOUR_PCT FIVE_HOUR_RESET SEVEN_DAY_PCT SEVEN_DAY_RESET \
            EXTRA_ENABLED EXTRA_PCT EXTRA_USED EXTRA_LIMIT <<< "$parsed"
        FIVE_HOUR_PCT="${FIVE_HOUR_PCT:-0}"
        SEVEN_DAY_PCT="${SEVEN_DAY_PCT:-0}"
        EXTRA_PCT="${EXTRA_PCT:-0}"
        HAVE_USAGE=true
    fi
fi

# --- 11. Hooks aggregation ---

HOOKS_LINE=""
if [ -d "$PROJECT_DIR/.claude" ]; then
    all_hooks=""
    for settings_file in "$PROJECT_DIR/.claude/settings.json" "$PROJECT_DIR/.claude/settings.local.json"; do
        [ -f "$settings_file" ] || continue
        file_hooks=$(jq -r '
            .hooks // {} | to_entries[] | .key as $ev |
            .value[].hooks[]?.command //empty |
            "\($ev)|\(split("/")[-1] | sub("\\.(sh|py|js|ts)$";""))"
        ' "$settings_file" 2>/dev/null) || file_hooks=""
        if [ -n "$file_hooks" ]; then
            all_hooks="${all_hooks}${all_hooks:+$'\n'}${file_hooks}"
        fi
    done

    if [ -n "$all_hooks" ]; then
        # Deduplicate and group by event
        HOOKS_LINE=$(printf '%s' "$all_hooks" | sort -u | awk -F'|' '
            { events[$1] = events[$1] ? events[$1] ", " $2 : $2 }
            END { for (ev in events) printf "%s[%s] ", ev, events[ev] }
        ')
        HOOKS_LINE="${HOOKS_LINE% }"  # trim trailing space
    fi
fi

# --- 12. Render output ---

# Line 1: Model (branch) | tokens | % used | % remain | thinking
used_color=$(color_for_pct "$CTX_PCT")
line1="${C_BLUE}${DISPLAY_NAME}${C_RESET}"
if [ -n "$GIT_BRANCH" ]; then
    line1="${line1} ${C_DIM}(${C_CYAN}${GIT_BRANCH}${C_DIM})${C_RESET}"
fi
if [ -n "$GIT_WORKTREE" ]; then
    line1="${line1} ${C_DIM}[${C_ORANGE}${GIT_WORKTREE}${C_DIM}]${C_RESET}"
fi
line1="${line1} ${C_DIM}|${C_RESET} "
line1="${line1}${C_ORANGE}${USED_FMT}/${TOTAL_FMT}${C_RESET}"
line1="${line1} ${C_DIM}|${C_RESET} "
line1="${line1}${used_color}${CTX_PCT}% used${C_RESET} ${C_ORANGE}${USED_COMMA}${C_RESET}"
line1="${line1} ${C_DIM}|${C_RESET} "
line1="${line1}${C_CYAN}${REMAIN_PCT}% remain${C_RESET} ${C_BLUE}${REMAIN_COMMA}${C_RESET}"
line1="${line1} ${C_DIM}|${C_RESET} "
line1="${line1}${C_WHITE}thinking:${C_RESET} ${THINKING_COLOR}${THINKING}${C_RESET}"

printf '%s' "$line1"

# Lines 2-3: Usage bars and reset times (only if API data available)
if [ "$HAVE_USAGE" = true ]; then
    sep=" ${C_DIM}|${C_RESET} "
    bar_width=10

    # Line 2: Usage bars
    five_bar=$(build_bar "$FIVE_HOUR_PCT" "$bar_width")
    seven_bar=$(build_bar "$SEVEN_DAY_PCT" "$bar_width")

    line2="${C_WHITE}current:${C_RESET} ${five_bar} ${C_CYAN}${FIVE_HOUR_PCT}%${C_RESET}"
    line2="${line2}${sep}"
    line2="${line2}${C_WHITE}weekly:${C_RESET} ${seven_bar} ${C_CYAN}${SEVEN_DAY_PCT}%${C_RESET}"

    if [ "$EXTRA_ENABLED" = "true" ]; then
        extra_bar=$(build_bar "$EXTRA_PCT" "$bar_width")
        extra_used_fmt=$(printf '%.2f' "$EXTRA_USED" 2>/dev/null || printf '%s' "$EXTRA_USED")
        extra_limit_fmt=$(printf '%.2f' "$EXTRA_LIMIT" 2>/dev/null || printf '%s' "$EXTRA_LIMIT")
        line2="${line2}${sep}"
        line2="${line2}${C_WHITE}extra:${C_RESET} ${extra_bar} ${C_CYAN}\$${extra_used_fmt}/\$${extra_limit_fmt}${C_RESET}"
    fi

    printf '\n%s' "$line2"

    # Line 3: Reset times
    five_reset=$(format_reset_time_relative "$FIVE_HOUR_RESET")
    seven_reset=$(format_reset_time_absolute "$SEVEN_DAY_RESET")

    if [ -n "$five_reset" ] || [ -n "$seven_reset" ]; then
        line3=""
        if [ -n "$five_reset" ]; then
            line3="${C_WHITE}resets${C_RESET} ${C_DIM}${five_reset}${C_RESET}"
        fi
        if [ -n "$seven_reset" ]; then
            [ -n "$line3" ] && line3="${line3}${sep}"
            line3="${line3}${C_WHITE}resets${C_RESET} ${C_DIM}${seven_reset}${C_RESET}"
        fi
        if [ "$EXTRA_ENABLED" = "true" ]; then
            # Extra resets on 1st of next month
            next_month_reset=$(date_calc "$(date +%Y-%m-01) +1 month" "%b %-d" 2>/dev/null | tr '[:upper:]' '[:lower:]')
            if [ -n "$next_month_reset" ]; then
                line3="${line3}${sep}${C_WHITE}resets${C_RESET} ${C_DIM}${next_month_reset}${C_RESET}"
            fi
        fi
        printf '\n%s' "$line3"
    fi
fi

# Line 4: Hooks overview
if [ -n "$HOOKS_LINE" ]; then
    printf '\n%s%s%s' "${C_DIM}hooks:${C_RESET} ${C_WHITE}" "$HOOKS_LINE" "${C_RESET}"
fi

# Set terminal window/tab title via OSC 2 escape sequence
# Format: "Claude: branch [worktree]" or "Claude: branch"
tab_title="Claude: ${GIT_BRANCH:-${DISPLAY_NAME}}"
if [ -n "$GIT_WORKTREE" ]; then
    tab_title="${tab_title} [${GIT_WORKTREE}]"
fi
printf '\033]2;%s\007' "$tab_title"
