#!/usr/bin/env python3
"""Demo script for RLM (Recursive Language Model).

Demonstrates using RLM with different backends:
- Anthropic API (Claude)
- OpenAI-compatible (Ollama, vLLM, etc.)
- Custom callback
"""

import argparse
import os
import sys
from pathlib import Path

from rlm import RLM
from rlm.backends import AnthropicBackend, CallbackBackend, OpenAICompatibleBackend


def mock_llm_callback(messages: list[dict[str, str]], model: str) -> str:
    """Mock LLM callback for testing without API access.

    Args:
        messages: List of messages
        model: Model identifier

    Returns:
        Mock response
    """
    # Simple mock that returns basic exploration code
    last_msg = messages[-1]["content"] if messages else ""

    if "Output:" in last_msg:
        # Second iteration - return final answer
        return """Based on the exploration, I'll provide the final answer:

```python
FINAL("This is a mock demo. In a real scenario, I would analyze the CONTEXT variable and provide insights based on your query.")
```
"""
    else:
        # First iteration - explore context
        return """I'll explore the CONTEXT to answer your query.

```python
print(f"Context size: {len(CONTEXT):,} characters")
sample = CONTEXT[:500]
print(f"Sample: {sample[:200]}...")
```
"""


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="RLM Demo")
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
        help="Path to context file (default: sample_data/large_document.txt)",
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

    args = parser.parse_args()

    # Load context
    if args.context_file:
        context_path = args.context_file
    else:
        context_path = Path(__file__).parent / "sample_data" / "large_document.txt"

    if not context_path.exists():
        print(f"Error: Context file not found: {context_path}", file=sys.stderr)
        print("\nCreating sample document...", file=sys.stderr)

        # Create sample document
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(
            """# Sample Document for RLM Testing

## Introduction
This is a sample document demonstrating the Recursive Language Model (RLM) pattern.
RLM allows processing of documents far exceeding typical LLM context windows.

## Overview
The RLM pattern treats long prompts as objects in an external, programmable environment.
Instead of inputting the entire document into the LLM's context window, the LLM:

1. Inspects the document structure
2. Searches for relevant sections
3. Chunks and processes recursively
4. Synthesizes the final answer

## Key Features

### Programmable Context
- CONTEXT variable holds the full document
- LLM writes Python code to explore it
- No need to fit everything in context window

### Recursive Processing
- Use llm_query() to process subsections
- Each recursive call handles manageable chunks
- Results are aggregated at the top level

### Security
- Sandboxed execution environment
- Blocked dangerous operations
- Restricted built-in functions

## Architecture

The RLM system consists of:

1. **RLM Orchestrator**: Manages iteration loop and parses code blocks
2. **Root LLM**: Receives query and generates exploration code
3. **REPL Environment**: Executes code with CONTEXT variable
4. **Recursion**: Sub-LLM calls via llm_query() function

## Benefits

### Handles Massive Documents
Process documents with millions of tokens, far beyond standard context windows.

### Adaptive Exploration
LLM decides how to explore based on the task, not a fixed retrieval strategy.

### Cost Efficient
Only processes relevant sections, reducing token usage compared to long-context models.

### Flexible
Works with any LLM backend - Anthropic, OpenAI, Ollama, local models, etc.

## Comparison to Other Approaches

### vs. RAG (Retrieval-Augmented Generation)
RAG retrieves and inserts passages. RLM treats the entire context as programmable,
deciding dynamically how to break down and explore the input.

### vs. Long-Context Models
Long-context models extend the window but still have limits. RLM is agnostic to
context size, handling arbitrarily large inputs through decomposition.

## Conclusion
RLM represents a new inference paradigm for language models, enabling processing
of contexts far beyond traditional limits through programmatic exploration and
recursive decomposition.

## References
- arXiv:2512.24601 - Recursive Language Models
- MIT CSAIL OASYS Lab
- Alex L. Zhang, Tim Kraska, and Omar Khattab
"""
        )

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
            backend = CallbackBackend(mock_llm_callback)
            print("Using mock callback backend")

        else:
            print(f"Error: Unknown backend: {args.backend}", file=sys.stderr)
            return 1

    except ImportError as e:
        print(f"Error initializing backend: {e}", file=sys.stderr)
        print("\nInstall required dependencies:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
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


if __name__ == "__main__":
    sys.exit(main())
