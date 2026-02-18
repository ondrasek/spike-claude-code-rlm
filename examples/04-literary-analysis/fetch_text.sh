#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/sherlock_holmes.txt"
URL="https://www.gutenberg.org/cache/epub/1661/pg1661.txt"

if [ -f "$OUTPUT" ]; then
    echo "Already downloaded: $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
    exit 0
fi

echo "Downloading 'The Adventures of Sherlock Holmes' from Project Gutenberg..."
if curl -fsSL "$URL" -o "$OUTPUT"; then
    echo "Saved to: $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
else
    echo "Error: Download failed. You can manually download from:"
    echo "  $URL"
    echo "and save it as: $OUTPUT"
    exit 1
fi
