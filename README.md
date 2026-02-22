# RLM - Recursive Language Model

A modern Python 3.11+ implementation of the Recursive Language Model paradigm from MIT CSAIL research ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)).

## What is RLM?

Unlike traditional RAG (Retrieval-Augmented Generation), RLM treats document context as an **external variable** in a Python REPL environment. The LLM doesn't see the full document - instead, it writes Python code to:

1. **Inspect** the document (`len(CONTEXT)`, `CONTEXT[:1000]`)
2. **Search** it (`re.findall(r'pattern', CONTEXT)`)
3. **Chunk** it and process recursively (`llm_query(f"Summarize: {chunk}")`)
4. **Synthesize** results and return a final answer (`FINAL(answer)`)

This approach enables processing of documents **far exceeding** typical context windows while maintaining adaptive, task-specific exploration strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                          │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    RLM Orchestrator                      │
│  • Manages iteration loop                                │
│  • Parses code blocks from LLM response                  │
│  • Detects FINAL() answers                               │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    Root LLM                              │
│  • Receives query + system prompt (NOT the context!)     │
│  • Generates Python code to explore CONTEXT              │
│  • Calls llm_query() for sub-processing                  │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    REPL Environment                      │
│  • CONTEXT variable (holds the massive text)             │
│  • llm_query(prompt) function for recursive calls        │
│  • FINAL(answer) / FINAL_VAR(var) for returning results  │
│  • Isolation delegated to container runtime               │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.11+ (tested with 3.13+)
- `anthropic>=0.39.0` (for Anthropic API backend)
- `openai>=1.0.0` (optional, for local models via Ollama/vLLM)

## Quick Start

### Using Anthropic API

```python
from rlm import RLM
from rlm.backends import AnthropicBackend

# Initialize with API key (or set ANTHROPIC_API_KEY env var)
backend = AnthropicBackend()
rlm = RLM(
    backend,
    model="claude-sonnet-4-20250514",
    recursive_model="claude-haiku-3-20250813",  # Cheaper model for sub-calls
    verbose=True,
)

# Process a large document
with open("large_document.txt") as f:
    context = f.read()

result = rlm.completion(
    context=context,
    query="What are the main themes discussed in this document?"
)

print(result.answer)
print(rlm.cost_summary())
```

### Using OpenAI

```bash
OPENAI_API_KEY=... uvx --with openai rlm --backend openai --context-file doc.txt --query "Summarize"
```

### Using OpenRouter

```bash
OPENROUTER_API_KEY=... uvx --with openai rlm --backend openrouter --context-file doc.txt --query "Summarize"
```

### Using Hugging Face

```bash
HF_TOKEN=... uvx --with openai rlm --backend huggingface --context-file doc.txt --query "Summarize"
```

### Using Local Models (Ollama)

```python
from rlm import RLM
from rlm.backends import OpenAICompatibleBackend

backend = OpenAICompatibleBackend(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
rlm = RLM(backend, model="llama3.2", verbose=True)

result = rlm.completion(context=doc, query="Summarize this document")
```

## Demo

```bash
# With Anthropic API (set ANTHROPIC_API_KEY)
python demo.py --verbose

# With Ollama
python demo.py --backend ollama --model llama3.2 --verbose

# Custom context file
python demo.py --context-file /path/to/document.txt --query "Your question here"
```

## API Reference

### RLM Class

```python
RLM(
    backend: LLMBackend,           # Backend instance
    model: str = "claude-sonnet-4-20250514",  # Root LLM model
    recursive_model: str = None,   # Model for llm_query (defaults to model)
    max_iterations: int = 10,      # Max REPL iterations
    max_depth: int = 3,            # Max recursion depth for llm_query
    verbose: bool = False,         # Print debug output
    compact_prompt: bool = False,  # Use shorter system prompt
)
```

**Methods:**
- `completion(context: str, query: str) -> RLMResult` - Sync completion
- `acompletion(context: str, query: str) -> RLMResult` - Async completion
- `cost_summary() -> dict` - Get usage statistics

### RLMResult

```python
@dataclass
class RLMResult:
    answer: str              # Final answer
    stats: RLMStats          # Statistics
    history: list[dict]      # Iteration history
    success: bool            # Whether completion succeeded
    error: str | None        # Error message if failed
```

### Backends

| Backend | Use Case | Default Model |
|---------|----------|---------------|
| `AnthropicBackend` | Direct Anthropic API | `claude-sonnet-4-20250514` |
| `OpenAICompatibleBackend` | OpenAI, OpenRouter, Hugging Face, Ollama, vLLM, etc. | (varies by preset) |

#### CLI Backend Presets

| `--backend` | Provider | API Key Env Var | Default Model |
|---|---|---|---|
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| `openai` | OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4` |
| `huggingface` | Hugging Face | `HF_TOKEN` | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `ollama` | Ollama (local) | *(none)* | `qwen3-coder:32b` |
| `claude` | Claude CLI | *(none)* | `claude-sonnet-4-20250514` |

## REPL Environment

The REPL provides these to the LLM:

| Name | Type | Description |
|------|------|-------------|
| `CONTEXT` | str | The full document (never print directly!) |
| `llm_query(prompt)` | function | Call sub-LLM, returns string |
| `FINAL(answer)` | function | Set final answer and complete |
| `FINAL_VAR(var_name)` | function | Set variable as final answer |
| `re`, `json`, `math`, `collections`, `itertools` | modules | Pre-imported |

**Isolation:** The REPL is designed to run inside a rootless container. No in-process sandboxing is applied.

## Project Structure

```
spike-claude-code-rlm/
├── rlm/
│   ├── __init__.py      # Package exports
│   ├── repl.py          # REPL environment (REPLEnv, REPLResult)
│   ├── backends.py      # LLM backends (Anthropic, OpenAI-compatible)
│   ├── prompts.py       # System prompts for root LLM
│   └── rlm.py           # Core orchestrator (RLM, RLMResult, RLMStats)
├── demo.py              # CLI demo with multiple backends
├── sample_data/
│   └── large_document.txt
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

## Features

✅ **Python 3.11+ Modern Implementation**
- Type hints with modern syntax (compatible with 3.13+)
- Dataclasses for clean data structures
- Async support (acompletion)

✅ **Multiple LLM Backends**
- Anthropic (Claude)
- OpenAI-compatible (Ollama, vLLM, etc.)
- Claude CLI backend

✅ **Container-Isolated Execution**
- Designed for rootless container runtimes
- No in-process sandbox overhead
- Full Python stdlib available to LLM-generated code

✅ **Recursive Processing**
- Configurable recursion depth
- Separate models for root/recursive calls
- Cost tracking and statistics

## References

- [arXiv Paper: Recursive Language Models](https://arxiv.org/abs/2512.24601)
- [Alex Zhang's Blog Post](https://alexzhang13.github.io/blog/2025/rlm/)
- [Official Implementation](https://github.com/alexzhang13/rlm)
- [MIT CSAIL OASYS Lab](https://github.com/alexzhang13/rlm)

## License

BSD 2-Clause License - See [LICENSE](LICENSE) file for details.

Based on research by Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL).
