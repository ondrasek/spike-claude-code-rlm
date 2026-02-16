#!/usr/bin/env bash
set -e

# Build and start the devcontainer
# Config passed via temp file (deleted immediately after reading in container)

# Navigate to repo root if in .devcontainer dir
[ -f ./devcontainer.json ] && cd ..

# Check GitHub CLI authentication
echo "Checking GitHub CLI authentication..."
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI is not authenticated."
    echo "   Please run 'gh auth login' first, then retry."
    exit 1
fi
echo "GitHub CLI authenticated"

# Gather configuration from host
ghToken=$(gh auth token)
gitUserName=$(git config --global user.name || echo "")
gitUserEmail=$(git config --global user.email || echo "")
gitSigningKey=$(git config --global user.signingkey 2>/dev/null || echo "")
gitGpgSign=$(git config --global commit.gpgsign 2>/dev/null || echo "")
# Get GitHub repository in owner/name format for cloning in container
githubRepo=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "")

echo "Environment for postCreate:"
echo "  githubRepo: $githubRepo"
echo "  gitUserName: $gitUserName"
echo "  gitUserEmail: $gitUserEmail"
if [ -n "$gitSigningKey" ]; then
    echo "  gitSigningKey: $gitSigningKey"
    echo "  gitGpgSign: $gitGpgSign"
else
    echo "  gitSigningKey: (not configured)"
fi
echo "  GH_TOKEN: (set)"
echo

# Write config to temp file (read by postCreate.sh)
# Note: File persists in .devcontainer/config/ but is gitignored
configFile=.devcontainer/config/postCreate.env.tmp
cat > "$configFile" << EOF
GH_TOKEN="$ghToken"
GITHUB_REPOSITORY="$githubRepo"
GIT_USER_NAME="$gitUserName"
GIT_USER_EMAIL="$gitUserEmail"
GIT_SIGNING_KEY="$gitSigningKey"
GIT_GPG_SIGN="$gitGpgSign"
EOF

# Build with cache management
echo "Building DevContainer..."
if ! devcontainer build --workspace-folder .; then
    echo "Build failed, retrying without cache..."
    devcontainer build --workspace-folder . --no-cache
fi

# Start container
devcontainer up --workspace-folder . --remove-existing-container
