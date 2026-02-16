#!/usr/bin/env bash
set -e

# Fix workspace permissions
sudo chown -R vscode:vscode /workspaces

# Load configuration from temp file (created by build.sh, cleaned up by build.sh after container starts)
configFile=/tmp/.devcontainer/config/postCreate.env.tmp
if [ -f "$configFile" ]; then
    set -a  # Export all variables
    source "$configFile"
    set +a
else
    echo "Warning: Config file not found. Run build.sh to start the devcontainer."
    echo "   Missing: $configFile"
fi

# REPO_NAME is set in Dockerfile from devcontainer.json build.args
workingCopy=/workspaces/${REPO_NAME}

echo "Setting up devcontainer..."
echo "   Repository: ${GITHUB_REPOSITORY:-$REPO_NAME}"
echo "   Git user: $GIT_USER_NAME <$GIT_USER_EMAIL>"

# Detect runtime environment
if [ "$CODESPACES" = "true" ]; then
    export RUNTIME_ENV="codespaces"
    echo "GitHub Codespaces environment detected"
else
    export RUNTIME_ENV="devcontainer"
    echo "DevContainer environment detected"
fi

# Configure Git user (static config is in Dockerfile)
echo "Configuring Git..."
if [ "$CODESPACES" != "true" ]; then
    if [ -z "$(git config --global user.name)" ]; then
        git config --global user.name "$GIT_USER_NAME"
        git config --global user.email "$GIT_USER_EMAIL"
    fi
    # Configure GPG commit signing if host has it configured
    if [ -n "$GIT_SIGNING_KEY" ]; then
        git config --global user.signingkey "$GIT_SIGNING_KEY"
        echo "   GPG signing key: $GIT_SIGNING_KEY"
        if [ "$GIT_GPG_SIGN" = "true" ]; then
            git config --global commit.gpgsign true
            echo "   GPG commit signing: enabled"
        fi
    fi
fi
git config --global --add safe.directory $workingCopy

# Setup GitHub authentication
if [ "$CODESPACES" != "true" ]; then
    echo "Setting up GitHub CLI authentication..."
    if [ -n "$GH_TOKEN" ]; then
        # GH_TOKEN is set, gh CLI will use it automatically
        echo "GitHub CLI will use GH_TOKEN from environment"
    else
        echo "Warning: No GH_TOKEN found - run 'gh auth login' after setup"
    fi
fi

# Clone repository (DevContainer only)
if [ "$CODESPACES" != "true" ]; then
    echo "Setting up repository..."
    if [ -d $workingCopy/.git ]; then
        cd $workingCopy && gh repo sync && cd -
        echo "Repository synced"
    else
        gh repo clone "$GITHUB_REPOSITORY" $workingCopy
        echo "Repository cloned"
    fi
fi

# Install dependencies and pre-commit hooks
echo "Installing dependencies..."
cd $workingCopy
uv sync
uv run pre-commit install
echo "Dependencies and pre-commit hooks installed"

# Configure Claude Code local settings (hooks for devcontainer only)
echo "Configuring Claude Code local settings..."
claudeDir="$workingCopy/.claude"
claudeLocalSettings="$claudeDir/settings.local.json"

# Ensure .claude directory exists (should exist from repo, but be safe)
mkdir -p "$claudeDir"

# Always create/overwrite settings.local.json since it's gitignored.
# statusline.sh and hooks live in the repo under .claude/ — no installation needed.
# $CLAUDE_PROJECT_DIR is resolved by Claude Code at runtime (not by the shell).
cat > "$claudeLocalSettings" << 'SETTINGS'
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "model": "opus",
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "statusLine": {
    "type": "command",
    "command": "$CLAUDE_PROJECT_DIR/.claude/statusline.sh"
  },
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/auto-commit.sh",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
SETTINGS
echo "Claude Code local settings created"

# Configure opencode
echo "Configuring opencode..."
# Symlink CLAUDE.md as AGENTS.md for opencode (reads project instructions from AGENTS.md)
if [ -f "$workingCopy/CLAUDE.md" ] && [ ! -f "$workingCopy/AGENTS.md" ]; then
    ln -sf CLAUDE.md "$workingCopy/AGENTS.md"
    echo "AGENTS.md symlinked to CLAUDE.md for opencode"
fi
# Verify opencode plugins are present (shipped in .opencode/plugins/ in the repo)
if [ -d "$workingCopy/.opencode/plugins" ]; then
    plugin_count=$(ls "$workingCopy/.opencode/plugins/"*.ts 2>/dev/null | wc -l)
    echo "opencode plugins installed ($plugin_count plugins: statusline, stop-hook)"
else
    echo "Warning: opencode plugins directory not found at $workingCopy/.opencode/plugins/"
fi
echo "opencode configured (run 'opencode' then '/connect' to authenticate providers)"

# Create worktrees directory for multi-branch development
mkdir -p /workspaces/worktrees
echo "Worktrees directory created"

# Verify installations
echo ""
echo "Verifying installations..."
echo "   Python: $(python3 --version)"
echo "   uv: $(uv --version)"
echo "   Git: $(git --version)"
echo "   gh: $(gh --version | head -1)"
echo "   Claude CLI: $(claude --version 2>/dev/null || echo 'installed (requires API key)')"
echo "   opencode: $(opencode version 2>/dev/null || echo 'installed (run opencode to start)')"
echo ""
echo "DevContainer setup completed!"
