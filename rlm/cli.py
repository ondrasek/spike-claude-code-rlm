"""CLI entry point for RLM (Recursive Language Model).

Provides a command-line interface for running RLM with different backends:
- Anthropic API (Claude)
- OpenAI-compatible (Ollama, vLLM, etc.)
- Custom callback (mock, for testing)

Usage:
    uvx rlm --context-file document.txt --query "What are the main themes?"
    uvx rlm --backend ollama --model llama3.2 --context-file doc.txt --query "Summarize"
    uvx rlm --backend callback --verbose  # Mock backend for testing
"""

import argparse
import os
import sys
from importlib import resources
from pathlib import Path

from .backends import (
    AnthropicBackend,
    CallbackBackend,
    ClaudeCLIBackend,
    LLMBackend,
    OpenAICompatibleBackend,
)
from .context import CompositeContext
from .rlm import RLM


def _mock_llm_callback(messages: list[dict[str, str]], model: str) -> str:
    """Mock LLM callback for testing without API access.

    Parameters
    ----------
    messages : list[dict[str, str]]
        List of messages.
    model : str
        Model identifier.

    Returns
    -------
    str
        Mock response.
    """
    last_msg = messages[-1]["content"] if messages else ""

    if "Output:" in last_msg:
        return """Based on the exploration, I'll provide the final answer:

```python
mock_answer = "This is a mock demo. In a real scenario, I would analyze "
mock_answer += "the CONTEXT variable and provide insights based on your query."
FINAL(mock_answer)
```
"""
    return """I'll explore the CONTEXT to answer your query.

```python
print(f"Context size: {len(CONTEXT):,} characters")
sample = CONTEXT[:500]
print(f"Sample: {sample[:200]}...")
```
"""


def _get_default_context() -> str:
    """Read the bundled sample document.

    Returns
    -------
    str
        Contents of the sample document.
    """
    return (resources.files("rlm") / "sample_data" / "large_document.txt").read_text(
        encoding="utf-8"
    )


def _load_context(args: argparse.Namespace) -> str | Path | CompositeContext:
    """Load context based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    str | Path | CompositeContext
        Loaded context.

    Raises
    ------
    FileNotFoundError
        If a specified context file or directory doesn't exist.
    NotADirectoryError
        If --context-dir doesn't point to a directory.
    """
    if args.context_dir:
        context_dir: Path = args.context_dir
        if not context_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {context_dir}")
        context = CompositeContext.from_directory(context_dir, glob=args.context_glob)
        print(
            f"Context: {len(context.files)} files from {context_dir} ({len(context):,} bytes total)"
        )
        return context

    if args.context_files and len(args.context_files) > 1:
        for p in args.context_files:
            if not p.exists():
                raise FileNotFoundError(f"Context file not found: {p}")
        context = CompositeContext.from_paths(args.context_files)
        print(f"Context: {len(context.files)} files ({len(context):,} bytes total)")
        return context

    if args.context_files:
        context_path: Path = args.context_files[0]
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")
        file_size = context_path.stat().st_size
        print(f"Context file: {context_path} ({file_size:,} bytes, memory-mapped)")
        return context_path

    context_str = _get_default_context()
    print(f"Loaded context: {len(context_str):,} characters from bundled sample")
    return context_str


def _resolve_ollama_url(base_url_override: str | None) -> str:
    """Resolve the Ollama base URL from CLI flag or environment.

    Parameters
    ----------
    base_url_override : str | None
        Explicit ``--base-url`` value (highest priority).

    Returns
    -------
    str
        Fully-qualified OpenAI-compatible base URL.
    """
    if base_url_override:
        return base_url_override
    ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    return f"{ollama_host.rstrip('/')}/v1"


def _create_backend(args: argparse.Namespace) -> LLMBackend:
    """Create LLM backend based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    LLMBackend
        Configured backend instance.

    Raises
    ------
    ValueError
        If ANTHROPIC_API_KEY is missing for anthropic backend.
    """
    if args.backend == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        print(f"Using Anthropic backend with model: {args.model}")
        return AnthropicBackend()

    if args.backend == "ollama":
        base_url = _resolve_ollama_url(args.base_url)
        print(f"Using Ollama backend with model: {args.model}")
        print(f"  Base URL: {base_url}")
        return OpenAICompatibleBackend(base_url=base_url, api_key="ollama")

    if args.backend == "claude":
        print(f"Using Claude CLI backend with model: {args.model}")
        return ClaudeCLIBackend()

    if args.backend == "callback":
        print("Using mock callback backend")
        return CallbackBackend(_mock_llm_callback)

    raise ValueError(f"Unknown backend: {args.backend}")


def _run_completion(rlm: RLM, context: str | Path | CompositeContext, query: str) -> int:
    """Execute the RLM completion loop and print results.

    Parameters
    ----------
    rlm : RLM
        Configured RLM instance.
    context : str | Path | CompositeContext
        Loaded context.
    query : str
        User query.

    Returns
    -------
    int
        Exit code.
    """
    print("=" * 80)
    print("Starting RLM completion...")
    print("=" * 80 + "\n")

    try:
        result = rlm.completion(context=context, query=query)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError during completion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    if not result.success:
        print(f"\nError: {result.error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result.answer)
    print("\n" + "=" * 80)
    print("STATISTICS:")
    print("=" * 80)
    stats = rlm.cost_summary()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="RLM - Recursive Language Model for processing long contexts",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama", "claude", "callback"],
        default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model identifier (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for OpenAI-compatible backends (overrides OLLAMA_HOST env var)",
    )
    parser.add_argument(
        "--recursive-model",
        help="Model for recursive calls (defaults to --model)",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        action="append",
        dest="context_files",
        help="Path to context file (repeatable for multi-file context)",
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        help="Directory of files to use as context (all files recursively)",
    )
    parser.add_argument(
        "--context-glob",
        default="**/*",
        help="Glob pattern when using --context-dir (default: '**/*')",
    )
    parser.add_argument(
        "--query",
        default="What are the main topics discussed in this document?",
        help="Query to ask about the context",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact system prompt",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum REPL iterations (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per LLM response (default: 4096)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    args = parser.parse_args()

    # Load context — files are memory-mapped so contexts larger than RAM work.
    try:
        context = _load_context(args)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading context: {e}", file=sys.stderr)
        return 1

    print(f"Query: {args.query}\n")

    # Initialize backend
    try:
        backend = _create_backend(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Error initializing backend: {e}", file=sys.stderr)
        print("\nInstall required dependencies:", file=sys.stderr)
        print("  pip install anthropic  # or: pip install openai", file=sys.stderr)
        return 1

    # Create RLM instance
    rlm = RLM(
        backend=backend,
        model=args.model,
        recursive_model=args.recursive_model,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        compact_prompt=args.compact,
    )

    return _run_completion(rlm, context, args.query)


def _get_version() -> str:
    """Get the package version.

    Returns
    -------
    str
        Version string.
    """
    from . import __version__

    return __version__
