---
name: rlm
description: Analyze a large document using the Recursive Language Model (RLM) pattern. Use when the user wants to process, summarize, or query a document that may exceed typical context windows, or when they ask to "run rlm" or "analyze a document with rlm".
argument-hint: [context-file] [query]
allowed-tools: Bash(uv run *), Read, Glob
---

# RLM - Recursive Language Model

Run the bundled RLM tool to analyze a document by recursively exploring it with LLM-generated Python code in a REPL.

## How to run

The RLM Python package is bundled in this plugin. Run it with `uv run`:

```bash
uv run --directory "${CLAUDE_PLUGIN_ROOT}" rlm --context-file <path> --query "<question>" --verbose
```

This requires no pre-installation — `uv` resolves dependencies and executes from the bundled source automatically.

### Arguments

Parse `$ARGUMENTS` to determine the context file and query:
- First argument (`$0`): path to the context file
- Remaining arguments: the query (join them as a single string)

If only one argument is provided, treat it as the context file and use a default query.
If no arguments are provided, ask the user which file to analyze and what to ask about it.

### Backend options

- **Anthropic** (default): Requires `ANTHROPIC_API_KEY` env var to be set.
  ```bash
  uv run --directory "${CLAUDE_PLUGIN_ROOT}" rlm --context-file <path> --query "<question>" --verbose
  ```

- **Ollama** (local models): Requires Ollama running locally.
  ```bash
  uv run --directory "${CLAUDE_PLUGIN_ROOT}" --with openai rlm --backend ollama --model llama3.2 --context-file <path> --query "<question>" --verbose
  ```

- **Mock/test**: No API key needed — useful for verifying the tool works.
  ```bash
  uv run --directory "${CLAUDE_PLUGIN_ROOT}" rlm --backend callback --verbose
  ```

### Additional flags

- `--model <name>`: Override the LLM model (default: claude-sonnet-4-20250514)
- `--recursive-model <name>`: Use a different (cheaper) model for recursive sub-queries
- `--compact`: Use a shorter system prompt
- `--max-iterations <n>`: Limit REPL iteration count (default: 10)

## Steps

1. Confirm the context file exists using the Read or Glob tool
2. Determine the appropriate backend based on available environment (check if `ANTHROPIC_API_KEY` is set)
3. Run the `uv run --directory "${CLAUDE_PLUGIN_ROOT}" rlm` command with the appropriate arguments
4. Present the final answer and statistics to the user
5. If the command fails, check the error output and suggest fixes (missing API key, file not found, etc.)
