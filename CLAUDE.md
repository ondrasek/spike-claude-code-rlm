# CLAUDE.md

## Project Overview

RLM (Recursive Language Model) is a Python 3.11+ implementation of the paradigm described in MIT CSAIL research paper [arXiv:2512.24601](https://arxiv.org/pdf/2512.24601). Unlike traditional RAG, RLM treats document context as an external variable in a sandboxed Python REPL, allowing LLMs to programmatically inspect, search, chunk, and recursively process documents that far exceed typical context windows.

**Status:** Alpha (v0.1.0)
**License:** BSD 2-Clause

## Quick Start

The tool is runnable via `uvx` (no install required):

```bash
# Run with mock backend (no API key needed, good for testing)
uvx rlm --backend callback --verbose

# Run with Anthropic
ANTHROPIC_API_KEY=... uvx rlm --context-file document.txt --query "Summarize this"

# Run with Ollama (requires openai extra)
uvx --with openai rlm --backend ollama --model llama3.2 --context-file doc.txt --query "Main points?"

# Also works as a module
python -m rlm --backend callback --verbose
```

## Repository Structure

```
spike-claude-code-rlm/
├── rlm/                        # Core package
│   ├── __init__.py             # Public API exports and version
│   ├── __main__.py             # python -m rlm support
│   ├── cli.py                  # CLI entry point (uvx rlm / python -m rlm)
│   ├── rlm.py                  # RLM orchestrator (iteration loop, code extraction)
│   ├── backends.py             # LLM backend implementations (Anthropic, OpenAI-compat, Callback)
│   ├── repl.py                 # Sandboxed REPL environment with security restrictions
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
│   ├── rlm/                    # Bundled Python package source
│   ├── pyproject.toml          # Build config for uv run from plugin dir
│   ├── requirements.txt        # Runtime dependencies
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

The system follows a loop: **User Query -> RLM Orchestrator -> LLM Backend -> Code Extraction -> Sandboxed REPL -> Output back to LLM -> repeat until FINAL()**.

### Core Components

- **`RLM`** (`rlm/rlm.py`): Orchestrator that manages the iteration loop (up to `max_iterations`), extracts Python code blocks from LLM markdown responses, executes them in the REPL, and feeds output back to the LLM. Key dataclasses: `RLMStats`, `RLMResult`.
- **`LLMBackend`** (`rlm/backends.py`): Abstract base class with three implementations:
  - `AnthropicBackend` — Direct Anthropic API (requires `anthropic` package, `ANTHROPIC_API_KEY` env var)
  - `OpenAICompatibleBackend` — For Ollama, vLLM, LM Studio (requires `openai` package, default URL `http://localhost:11434/v1`)
  - `CallbackBackend` — Wraps a `Callable[[list[dict], str], str]` for custom integrations
- **`REPLEnv`** (`rlm/repl.py`): Sandboxed execution environment providing `CONTEXT`, `llm_query()`, `FINAL()`, `FINAL_VAR()`, pre-imported modules (`re`, `json`, `math`, `collections`, `itertools`), and safe builtins. Blocks dangerous patterns (os/sys/subprocess imports, eval, exec, open, getattr, etc.).
- **`prompts.py`** (`rlm/prompts.py`): Two system prompts (`FULL_SYSTEM_PROMPT` at ~120 lines, `COMPACT_SYSTEM_PROMPT` at ~25 lines) that instruct the LLM on the inspect-search-chunk-synthesize strategy.
- **`cli.py`** (`rlm/cli.py`): CLI entry point registered as `[project.scripts] rlm = "rlm.cli:main"`. Provides argparse-based interface with `--backend`, `--model`, `--context-file`, `--query`, `--verbose`, `--compact`, `--max-iterations`, and `--version` flags.

## Tech Stack and Dependencies

- **Python:** >=3.11 (target: 3.13)
- **Build backend:** `hatchling`
- **Required:** `anthropic>=0.39.0`
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

# Run with mock backend (no API key needed)
uvx rlm --backend callback --verbose

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

There is no formal test suite yet. Testing is done manually:

```bash
uvx rlm --backend callback --verbose
```

The `CallbackBackend` with `_mock_llm_callback` in `rlm/cli.py` provides a deterministic mock that exercises the full iteration loop without requiring API access. Future testing should use `pytest`.

## Key Design Patterns

- **Strategy:** Multiple LLM backends behind abstract `LLMBackend` ABC
- **Factory:** `_create_llm_query_fn()` creates closures for recursive LLM calls with depth tracking
- **Sandbox:** REPL restricts `__builtins__` to `{}` and validates code against `BLOCKED_PATTERNS` before execution
- **Dataclasses:** `RLMStats`, `RLMResult`, `REPLResult` for clean data structures

## Important Implementation Details

- The default model is `claude-sonnet-4-20250514` (set in both `RLM.__init__` and `cli.py`)
- `AnthropicBackend` separates system messages from chat messages per Anthropic API requirements
- The REPL captures `print()` output (max 10,000 chars) and feeds it back to the LLM as iteration context
- `llm_query()` calls are limited by `max_depth` (default 3) to prevent infinite recursion
- The main loop runs up to `max_iterations` (default 10) attempts before failing
- Async methods (`acompletion`) currently delegate to their sync counterparts
- Sample data is bundled inside the package at `rlm/sample_data/` and accessed via `importlib.resources`

## Common Pitfalls

- Do not add new imports to `repl.py`'s REPL namespace without also updating `BLOCKED_PATTERNS` to account for security implications
- The `_extract_code_blocks` regex expects markdown code fences (`` ```python `` or `` ``` ``); changes to code block parsing affect the entire iteration loop
- `AnthropicBackend` creates a new `AsyncAnthropic` client on every `acompletion` call — this is by design for simplicity but could be optimized
- The `FINAL_VAR` implementation in `repl.py` is incomplete — `_final_var` is a no-op; the fallback logic checks for `final_` prefixed variables in the namespace instead
- When using `uvx` with the Ollama backend, pass `--with openai` since `openai` is an optional dependency

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For Anthropic backend | Anthropic API authentication key |

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
- `rlm/` — bundled Python package source
- `pyproject.toml` + `requirements.txt` — so `uv run` can build and execute from source

**Prerequisites:** `uv` on PATH and Python 3.11+. No other pre-installation needed.

**Keeping the plugin in sync:** When modifying Python source in `rlm/`, remember to also update `plugin/rlm/` (or automate the copy). The plugin bundles a snapshot of the package.

## Areas for Contribution

Per CONTRIBUTING.md: additional LLM backends, performance optimizations, error handling improvements, enhanced REPL security, pytest test suite, CI/CD pipeline, documentation, and example use cases.
