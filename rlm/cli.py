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

from .backends import AnthropicBackend, CallbackBackend, OpenAICompatibleBackend
from .rlm import RLM


def _mock_llm_callback(messages: list[dict[str, str]], model: str) -> str:
    """Mock LLM callback for testing without API access.

    Args:
        messages: List of messages
        model: Model identifier

    Returns:
        Mock response
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
    else:
        return """I'll explore the CONTEXT to answer your query.

```python
print(f"Context size: {len(CONTEXT):,} characters")
sample = CONTEXT[:500]
print(f"Sample: {sample[:200]}...")
```
"""


def _get_default_context_path() -> Path:
    """Get path to the bundled sample document.

    Returns:
        Path to the sample document
    """
    ref = resources.files("rlm") / "sample_data" / "large_document.txt"
    with resources.as_file(ref) as path:
        return Path(path)


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="RLM - Recursive Language Model for processing long contexts",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama", "callback"],
        default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model identifier (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--recursive-model",
        help="Model for recursive calls (defaults to --model)",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Path to context file (default: bundled sample document)",
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
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    args = parser.parse_args()

    # Load context
    context_path = args.context_file or _get_default_context_path()

    if not context_path.exists():
        print(f"Error: Context file not found: {context_path}", file=sys.stderr)
        return 1

    try:
        context = context_path.read_text()
    except Exception as e:
        print(f"Error reading context file: {e}", file=sys.stderr)
        return 1

    print(f"Loaded context: {len(context):,} characters from {context_path}")
    print(f"Query: {args.query}\n")

    # Initialize backend
    try:
        if args.backend == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                print(
                    "Error: ANTHROPIC_API_KEY environment variable not set",
                    file=sys.stderr,
                )
                return 1
            backend = AnthropicBackend()
            print(f"Using Anthropic backend with model: {args.model}")

        elif args.backend == "ollama":
            backend = OpenAICompatibleBackend(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            print(f"Using Ollama backend with model: {args.model}")

        elif args.backend == "callback":
            backend = CallbackBackend(_mock_llm_callback)
            print("Using mock callback backend")

        else:
            print(f"Error: Unknown backend: {args.backend}", file=sys.stderr)
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
        verbose=args.verbose,
        compact_prompt=args.compact,
    )

    # Run completion
    print("=" * 80)
    print("Starting RLM completion...")
    print("=" * 80 + "\n")

    try:
        result = rlm.completion(context=context, query=args.query)

        if result.success:
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
        else:
            print(f"\nError: {result.error}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\nError during completion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def _get_version() -> str:
    """Get the package version.

    Returns:
        Version string
    """
    from . import __version__

    return __version__
