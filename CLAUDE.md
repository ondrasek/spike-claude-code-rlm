# CLAUDE.md

## Project Overview

RLM (Recursive Language Model) is a Python 3.11+ implementation of the paradigm described in MIT CSAIL research paper [arXiv:2512.24601](https://arxiv.org/pdf/2512.24601) (see also the [blog post](https://alexzhang13.github.io/blog/2025/rlm/)). Unlike traditional RAG, RLM treats document context as an external variable in a Python REPL, allowing LLMs to programmatically inspect, search, chunk, and recursively process documents that far exceed typical context windows. The REPL is designed to run inside a rootless container for isolation.

**Status:** Alpha (v0.1.0)
**License:** Apache 2.0

## Quick Start

The tool is runnable via `uvx` (no install required):

```bash
# Run with Anthropic
ANTHROPIC_API_KEY=... uvx rlm --model claude-sonnet-4-20250514 --context-file document.txt --query "Summarize this"

# Run with OpenAI
OPENAI_API_KEY=... uvx --with openai rlm --backend openai --model gpt-4o --context-file doc.txt --query "Main points?"

# Run with OpenRouter
OPENROUTER_API_KEY=... uvx --with openai rlm --backend openrouter --model anthropic/claude-sonnet-4 --context-file doc.txt --query "Main points?"

# Run with Hugging Face
HF_TOKEN=... uvx --with openai rlm --backend huggingface --model Qwen/Qwen2.5-Coder-32B-Instruct --context-file doc.txt --query "Main points?"

# Run with Ollama (requires openai extra)
uvx --with openai rlm --backend ollama --model llama3.2 --context-file doc.txt --query "Main points?"

# Also works as a module
python -m rlm --backend ollama --model llama3.2 --context-file doc.txt --query "Summarize"
```

## Repository Structure

```
spike-claude-code-rlm/
├── rlm/                        # Core package
│   ├── __init__.py             # Public API exports and version
│   ├── __main__.py             # python -m rlm support
│   ├── cli.py                  # CLI entry point (uvx rlm / python -m rlm)
│   ├── config.py               # YAML config loading, validation, per-role resolution
│   ├── rlm.py                  # RLM orchestrator (iteration loop, code extraction)
│   ├── backends.py             # LLM backend implementations (Anthropic, OpenAI-compat)
│   ├── repl.py                 # REPL environment (container-isolated)
│   ├── prompts.py              # System prompts (full and compact) for LLM guidance
│   └── sample_data/
│       └── large_document.txt  # Bundled sample document for testing
├── .claude-plugin/
│   └── marketplace.json        # Marketplace catalog for plugin distribution
├── plugin/                     # Self-contained Claude Code plugin
│   ├── .claude-plugin/
│   │   └── plugin.json         # Plugin manifest (name, version, description)
│   ├── skills/
│   │   └── rlm/
│   │       └── SKILL.md        # Plugin skill (uses CLAUDE_PLUGIN_ROOT)
│   ├── rlm -> ../rlm           # Symlink to root package (single source of truth)
│   ├── pyproject.toml -> ../pyproject.toml
│   ├── requirements.txt -> ../requirements.txt
│   └── README.md               # Plugin installation instructions
├── demo.py                     # Convenience wrapper (delegates to rlm.cli)
├── examples.py                 # 10 example usage scenarios
├── pyproject.toml              # Build config (hatchling), dependencies, ruff/mypy settings
├── requirements.txt            # Runtime dependencies
├── README.md                   # User-facing documentation
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # BSD 2-Clause license
└── .github/
    └── copilot-instructions.md # AI coding guidelines
```

## Architecture

The system follows a loop: **User Query -> RLM Orchestrator -> LLM Backend -> Code Extraction -> REPL -> Output back to LLM -> repeat until FINAL()**.

### Core Components

- **`RLM`** (`rlm/rlm.py`): Orchestrator that manages the iteration loop (up to `max_iterations`), extracts Python code blocks from LLM markdown responses, executes them in the REPL, and feeds output back to the LLM. Key dataclasses: `RLMStats`, `RLMResult`.
- **`LLMBackend`** (`rlm/backends.py`): Abstract base class with three implementations:
  - `AnthropicBackend` — Direct Anthropic API (requires `anthropic` package, `ANTHROPIC_API_KEY` env var)
  - `OpenAICompatibleBackend` — For Ollama, vLLM, LM Studio (requires `openai` package, default URL `http://localhost:11434/v1`)
- **`REPLEnv`** (`rlm/repl.py`): Execution environment providing `CONTEXT` (a plain Python `str`), `FILES` dict (when multi-file), `SHOW_VARS()`, `llm_query(snippet, task)`, `FINAL()`, and pre-imported modules (`re`, `json`, `math`, `collections`, `itertools`). Context classes from `context.py` are used internally for file I/O but materialised to plain `str` before injection into the namespace. Isolation is delegated to the container runtime.
- **`prompts.py`** (`rlm/prompts.py`): System prompts for each LLM role — `FULL_SYSTEM_PROMPT` / `COMPACT_SYSTEM_PROMPT` for the root LM (inspect-search-chunk-synthesize strategy), `SUB_RLM_SYSTEM_PROMPT` for sub-RLM calls, and `VERIFIER_SYSTEM_PROMPT` for the optional `--verify` step.
- **`config.py`** (`rlm/config.py`): YAML configuration loader for per-role LLM settings. Dataclasses: `DefaultsConfig`, `RoleConfig`, `SettingsConfig`, `RLMConfig`, `ResolvedRoleConfig`. Functions: `load_config(path)` parses YAML, `resolve_role(name, config)` merges role > defaults and resolves env vars / prompt files. No `rlm.*` imports — consumed only by `cli.py`.
- **`cli.py`** (`rlm/cli.py`): CLI entry point registered as `[project.scripts] rlm = "rlm.cli:main"`. Provides argparse-based interface with `--backend` (anthropic, openai, openrouter, huggingface, ollama, claude), `--model`, `--config` (YAML config file for per-role settings), `--context-file`, `--query`, `--verbose`, `--compact`, `--max-iterations`, `--no-context-sample`, `--timeout`, `--max-token-budget`, `--verify`, and `--version` flags. The `_create_backend()` factory configures `OpenAICompatibleBackend` with the correct base URL and API key for each provider. When `--config` is provided, per-role backends are created via `_create_backend_from_resolved()` with merge priority: CLI flags > roles.{role} > defaults > hardcoded defaults.

## Tech Stack and Dependencies

- **Python:** >=3.11 (target: 3.13)
- **Build backend:** `hatchling`
- **Required:** `anthropic>=0.39.0`, `pyyaml>=6.0`
- **Optional extras:**
  - `ollama`: `openai>=1.0.0` (for OpenAI-compatible backends)
  - `dev`: `ruff>=0.8.0`, `mypy>=1.14.0`
- **Entry point:** `rlm = rlm.cli:main` (console script, enables `uvx rlm`)

## Development Commands

```bash
# Setup with uv
uv sync --extra dev

# Or traditional setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Linting
ruff check .

# Type checking
mypy rlm/

# Run with Anthropic
ANTHROPIC_API_KEY=... uvx rlm --verbose

# Run with Ollama
uvx --with openai rlm --backend ollama --model llama3.2 --verbose

# Build the package
uv build
```

## Code Style and Conventions

- **Formatting:** ruff with line length 100, target Python 3.13
- **Lint rules:** E, F, W, I (isort), N (naming), UP (pyupgrade), B (bugbear), A (builtins), C4 (comprehensions), SIM (simplify)
- **Type checking:** mypy strict mode — all functions must have type annotations, `disallow_untyped_defs: true`
- **Docstrings:** NumPy-style for all public modules, classes, and functions
- **Data structures:** Use `dataclasses` from stdlib (not Pydantic)
- **String formatting:** f-strings
- **Imports:** Relative imports within the package (e.g., `from .backends import LLMBackend`)
- **Error handling:** Specific exception types, informative messages, no bare `except:`
- **Functions:** Ideally under 50 lines, focused and single-purpose

## Testing

```bash
# Run the full test suite (excluding slow/Ollama tests)
uv run pytest -x --tb=short -m "not slow"

# Run Ollama smoke tests (requires running Ollama server)
OLLAMA_HOST=localhost:11434 uv run pytest -m ollama -v
```

## Key Design Patterns

- **Strategy:** Multiple LLM backends behind abstract `LLMBackend` ABC
- **Factory:** `_create_llm_query_fn()` creates closures for recursive LLM calls with depth tracking
- **Container isolation:** REPL delegates security to the rootless container runtime
- **Dataclasses:** `RLMStats`, `RLMResult`, `REPLResult` for clean data structures

## Important Implementation Details

- `--model` is required — there are no default models; the user must always specify one
- `AnthropicBackend` separates system messages from chat messages per Anthropic API requirements
- The REPL captures `print()` output (max 10,000 chars) and feeds it back to the LLM as iteration context
- `llm_query()` calls receive a `SUB_RLM_SYSTEM_PROMPT` (sub-RLM role) and are limited by `max_depth` (default 3) to prevent infinite recursion
- The main loop runs up to `max_iterations` (default 10) attempts before failing
- `--verify` runs an optional verification sub-call on the final answer using `VERIFIER_SYSTEM_PROMPT`
- Async methods (`acompletion`) currently delegate to their sync counterparts
- Sample data is bundled inside the package at `rlm/sample_data/` and accessed via `importlib.resources`

## Common Pitfalls

- `CONTEXT` in the REPL is a **plain Python `str`** — LLM-generated code should use standard Python (`re.findall(pattern, CONTEXT)`, `CONTEXT.splitlines()`, slicing) not custom methods. The context classes in `context.py` are internal file-loading infrastructure only.
- The `_extract_code_blocks` regex expects markdown code fences (`` ```python `` or `` ``` ``); changes to code block parsing affect the entire iteration loop
- `AnthropicBackend` creates a new `AsyncAnthropic` client on every `acompletion` call — this is by design for simplicity but could be optimized
- When using `uvx` with the Ollama backend, pass `--with openai` since `openai` is an optional dependency

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For Anthropic backend | Anthropic API authentication key |
| `OPENAI_API_KEY` | For OpenAI backend | OpenAI API authentication key |
| `OPENROUTER_API_KEY` | For OpenRouter backend | OpenRouter API authentication key |
| `HF_TOKEN` | For Hugging Face backend | Hugging Face API token |

## Claude Code Plugin

The `plugin/` directory is the distributable Claude Code plugin. It bundles the entire RLM Python package and skill so it can be installed independently — no PyPI publish required. The repo root contains a `.claude-plugin/marketplace.json` so the repository itself serves as a plugin marketplace.

### Installing the plugin

**From GitHub** (recommended for end users):
```
/plugin marketplace add ondrasek/spike-claude-code-rlm
/plugin install rlm@rlm-marketplace
```

**From a local clone:**
```
/plugin marketplace add ./path/to/spike-claude-code-rlm
/plugin install rlm@rlm-marketplace
```

**For development / one-off testing** (session-only, no install):
```bash
claude --plugin-dir ./plugin
```

**Team-wide via `.claude/settings.json`** (auto-prompts collaborators to install):
```json
{
  "extraKnownMarketplaces": {
    "rlm-marketplace": {
      "source": {
        "source": "github",
        "repo": "ondrasek/spike-claude-code-rlm"
      }
    }
  },
  "enabledPlugins": {
    "rlm@rlm-marketplace": true
  }
}
```

### Using the plugin

Once installed, the `/rlm:rlm` skill is available:
```
/rlm:rlm path/to/document.txt What are the main themes?
```
Claude also invokes the skill automatically when asked to analyze a large document with RLM.

### Uninstalling

```
/plugin uninstall rlm@rlm-marketplace
```

### Plugin internals

- `.claude-plugin/plugin.json` — manifest with name `rlm`, version, description
- `skills/rlm/SKILL.md` — skill that runs `uv run --directory "${CLAUDE_PLUGIN_ROOT}" rlm ...`
- `rlm/` — symlink to `../rlm` (single source of truth, no manual sync needed)
- `pyproject.toml` / `requirements.txt` — symlinks to root copies

**Prerequisites:** `uv` on PATH and Python 3.11+. No other pre-installation needed.

## Rules

- **Keep `ALGORITHM.md` up to date.** Whenever you change core implementation logic (the iteration loop, REPL environment, code extraction, prompt strategy, or backend behavior), update `ALGORITHM.md` to reflect the current state. This file documents how the implementation maps to the arXiv paper and is auto-converted to PDF by CI.

## Areas for Contribution

Per CONTRIBUTING.md: additional LLM backends, performance optimizations, error handling improvements, pytest test suite, CI/CD pipeline, documentation, and example use cases.
