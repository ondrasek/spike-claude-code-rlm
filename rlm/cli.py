"""CLI entry point for RLM (Recursive Language Model).

Provides a command-line interface for running RLM with different backends:
- Anthropic API (Claude)
- OpenAI (GPT-4o, etc.)
- OpenRouter (multi-provider gateway)
- Hugging Face Inference API
- OpenAI-compatible (Ollama, vLLM, etc.)

Usage:
    uvx rlm --context-file document.txt --query "What are the main themes?"
    uvx rlm --backend openai --context-file doc.txt --query "Summarize"
    uvx rlm --backend ollama --model llama3.2 --context-file doc.txt --query "Summarize"
"""

import argparse
import os
from importlib import resources
from pathlib import Path

import click

from .backends import (
    AnthropicBackend,
    ClaudeCLIBackend,
    LLMBackend,
    OpenAICompatibleBackend,
)
from .context import CompositeContext
from .rlm import RLM


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
        click.echo(
            f"Context: {len(context.files)} files from {context_dir} ({len(context):,} bytes total)"
        )
        return context

    if args.context_files and len(args.context_files) > 1:
        for p in args.context_files:
            if not p.exists():
                raise FileNotFoundError(f"Context file not found: {p}")
        context = CompositeContext.from_paths(args.context_files)
        click.echo(f"Context: {len(context.files)} files ({len(context):,} bytes total)")
        return context

    if args.context_files:
        context_path: Path = args.context_files[0]
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")
        file_size = context_path.stat().st_size
        click.echo(f"Context file: {context_path} ({file_size:,} bytes, memory-mapped)")
        return context_path

    context_str = _get_default_context()
    click.echo(f"Loaded context: {len(context_str):,} characters from bundled sample")
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


# Per-backend default models (used when --model is not specified).
_BACKEND_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-sonnet-4",
    "huggingface": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "ollama": "qwen3-coder:32b",
    "claude": "claude-sonnet-4-20250514",
}


def _resolve_model(args: argparse.Namespace) -> None:
    """Fill in the default model when the user didn't pass ``--model``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (mutated in place).
    """
    if args.model is None:
        args.model = _BACKEND_DEFAULT_MODELS.get(args.backend, "claude-sonnet-4-20250514")


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
        If a required API key environment variable is missing.
    """
    _resolve_model(args)

    if args.backend == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        click.echo(f"Using Anthropic backend with model: {args.model}")
        return AnthropicBackend()

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        base_url = args.base_url or "https://api.openai.com/v1"
        click.echo(f"Using OpenAI backend with model: {args.model}")
        click.echo(f"  Base URL: {base_url}")
        return OpenAICompatibleBackend(base_url=base_url, api_key=api_key)

    if args.backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        base_url = args.base_url or "https://openrouter.ai/api/v1"
        click.echo(f"Using OpenRouter backend with model: {args.model}")
        click.echo(f"  Base URL: {base_url}")
        return OpenAICompatibleBackend(base_url=base_url, api_key=api_key)

    if args.backend == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN environment variable not set")
        base_url = args.base_url or "https://api-inference.huggingface.co/v1"
        click.echo(f"Using Hugging Face backend with model: {args.model}")
        click.echo(f"  Base URL: {base_url}")
        return OpenAICompatibleBackend(base_url=base_url, api_key=api_key)

    if args.backend == "ollama":
        base_url = _resolve_ollama_url(args.base_url)
        click.echo(f"Using Ollama backend with model: {args.model}")
        click.echo(f"  Base URL: {base_url}")
        return OpenAICompatibleBackend(base_url=base_url, api_key="ollama")

    if args.backend == "claude":
        click.echo(f"Using Claude CLI backend with model: {args.model}")
        return ClaudeCLIBackend()

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
    click.echo(click.style("\U0001f680 STARTING RLM COMPLETION", bold=True, fg="cyan"))

    try:
        result = rlm.completion(context=context, query=query)
    except KeyboardInterrupt:
        click.echo(click.style("\n\U0000274c INTERRUPTED BY USER", bold=True, fg="red"), err=True)
        return 130
    except Exception as e:
        click.echo(click.style(f"\n\U0000274c ERROR: {e}", bold=True, fg="red"), err=True)
        import traceback

        traceback.print_exc()
        return 1

    if not result.success:
        click.echo(
            click.style(f"\n\U0000274c ERROR: {result.error}", bold=True, fg="red"), err=True
        )
        return 1

    click.echo(click.style("\n\U00002705 FINAL ANSWER", bold=True, fg="green"))
    click.echo(result.answer)

    click.echo(click.style("\n\U0001f4ca STATISTICS", bold=True, fg="blue"))
    stats = rlm.cost_summary()
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")
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
        choices=["anthropic", "openai", "openrouter", "huggingface", "ollama", "claude"],
        default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default depends on backend)",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL override for OpenAI-compatible backends",
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
        click.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        click.echo(f"Error loading context: {e}", err=True)
        return 1

    click.echo(f"Query: {args.query}\n")

    # Initialize backend
    try:
        backend = _create_backend(args)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return 1
    except ImportError as e:
        click.echo(f"Error initializing backend: {e}", err=True)
        click.echo("\nInstall required dependencies:", err=True)
        click.echo("  pip install anthropic  # or: pip install openai", err=True)
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
