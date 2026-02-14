# RLM Plugin for Claude Code

A self-contained Claude Code plugin that adds the `/rlm` skill for analyzing documents using the Recursive Language Model pattern.

## Installation

### From GitHub (recommended)

Add the marketplace and install the plugin:

```
/plugin marketplace add ondrasek/spike-claude-code-rlm
/plugin install rlm@rlm-marketplace
```

### From a local clone

If you have the repository cloned locally:

```
/plugin marketplace add ./path/to/spike-claude-code-rlm
/plugin install rlm@rlm-marketplace
```

### For development / one-off testing

Load the plugin directly for the current session without installing:

```bash
claude --plugin-dir ./plugin
```

## Usage

Once installed, use the `/rlm` skill (namespaced as `/rlm:rlm` if other plugins define an `rlm` command):

```
/rlm path/to/document.txt What are the main themes?
```

Claude will also invoke the skill automatically when you ask it to analyze a large document with RLM.

### Backend options

| Backend | Flag | Requirements |
|---|---|---|
| Anthropic (default) | `--backend anthropic` | `ANTHROPIC_API_KEY` env var |
| Ollama | `--backend ollama` | Ollama running locally |
| Mock/test | `--backend callback` | None |

### Prerequisites

- **uv** must be available on `PATH` (the plugin uses `uv run` to execute the bundled Python source)
- **Python 3.11+** must be installed

No other pre-installation is required. The plugin bundles the full RLM Python package and `uv` resolves dependencies automatically on first run.

## Uninstall

```
/plugin uninstall rlm@rlm-marketplace
```
