#!/usr/bin/env bash
set -e

# Start an existing devcontainer (use build.sh for fresh builds)

[ -f devcontainer.json ] && cd ..
devcontainer up --workspace-folder . "$@"
