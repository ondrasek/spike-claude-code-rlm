"""Example usage scenarios for RLM.

This file demonstrates various use cases and patterns for working with RLM.
Each example function can be run independently.
"""

from __future__ import annotations


# Example 1: Basic usage with Anthropic
def example_anthropic() -> None:
    """Example using Anthropic's Claude API."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(
        backend,
        model="claude-sonnet-4-20250514",
        verbose=True,
    )

    # Load a large document
    with open("large_document.txt") as f:
        context = f.read()

    result = rlm.completion(context=context, query="What are the key insights from this document?")

    print(result.answer)
    print(rlm.cost_summary())


# Example 2: Using different models for root and recursive calls
def example_tiered_models() -> None:
    """Example using expensive model for root, cheaper for recursive calls."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(
        backend,
        model="claude-sonnet-4-20250514",  # More capable for main reasoning
        recursive_model="claude-haiku-3-20250813",  # Faster/cheaper for sub-tasks
        verbose=True,
    )

    with open("document.txt") as f:
        context = f.read()

    result = rlm.completion(context=context, query="Summarize each section")
    print(result.answer)


# Example 3: Local models with Ollama
def example_ollama() -> None:
    """Example using local models via Ollama."""
    from rlm import RLM
    from rlm.backends import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require a real key
    )

    rlm = RLM(backend, model="llama3.2", verbose=True)

    with open("document.txt") as f:
        context = f.read()

    result = rlm.completion(context=context, query="Extract main points")
    print(result.answer)


# Example 4: Custom callback integration
def example_custom_callback() -> None:
    """Example integrating with a custom LLM system."""
    from rlm import RLM
    from rlm.backends import CallbackBackend

    def my_llm_function(messages: list[dict[str, str]], model: str) -> str:
        """Custom function that calls your LLM system.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Conversation history.
        model : str
            Model identifier.

        Returns
        -------
        str
            LLM response.
        """
        # Your custom integration here
        # Could call Claude Max, a CLI tool, or any other system
        return "Custom LLM response placeholder"

    backend = CallbackBackend(my_llm_function)
    rlm = RLM(backend, model="my-custom-model")

    result = rlm.completion(context="...", query="...")
    print(result.answer)


# Example 5: Processing structured data
def example_structured_data() -> None:
    """Example processing structured data like logs or JSON."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend, verbose=True)

    # Large JSON log file
    with open("server_logs.json") as f:
        context = f.read()

    result = rlm.completion(
        context=context,
        query="Find all errors in the last hour and group by error type",
    )

    print(result.answer)


# Example 6: Async usage
async def example_async() -> None:
    """Example using async completion."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend)

    with open("document.txt") as f:
        context = f.read()

    result = await rlm.acompletion(context=context, query="Analyze sentiment")
    print(result.answer)


# Example 7: Error handling
def example_error_handling() -> None:
    """Example demonstrating error handling."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend, max_iterations=5)

    with open("document.txt") as f:
        context = f.read()

    result = rlm.completion(context=context, query="Complex analysis")

    if result.success:
        print("Success!")
        print(result.answer)
    else:
        print(f"Error: {result.error}")
        # Inspect history to see what happened
        for item in result.history:
            print(f"Iteration {item['iteration']}: {item.get('output', 'N/A')}")


# Example 8: Analyzing code repositories
def example_code_analysis() -> None:
    """Example analyzing a code repository."""
    from pathlib import Path

    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend, verbose=True)

    # Concatenate all Python files
    repo_path = Path("my_project")
    files = []
    for py_file in repo_path.rglob("*.py"):
        content = py_file.read_text()
        files.append(f"=== {py_file} ===\n{content}\n")

    context = "\n".join(files)

    result = rlm.completion(context=context, query="What are the main architectural patterns used?")

    print(result.answer)


# Example 9: Comparing documents
def example_document_comparison() -> None:
    """Example comparing two large documents."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend)

    with open("doc1.txt") as f1, open("doc2.txt") as f2:
        doc1 = f1.read()
        doc2 = f2.read()

    # Combine documents with markers
    context = f"=== DOCUMENT 1 ===\n{doc1}\n\n=== DOCUMENT 2 ===\n{doc2}"

    result = rlm.completion(
        context=context, query="What are the key differences between these documents?"
    )

    print(result.answer)


# Example 10: Research paper analysis
def example_research_paper() -> None:
    """Example analyzing a research paper."""
    from rlm import RLM
    from rlm.backends import AnthropicBackend

    backend = AnthropicBackend()
    rlm = RLM(backend, verbose=True)

    with open("paper.txt") as f:
        context = f.read()

    queries = [
        "What is the main contribution of this paper?",
        "What are the key experimental results?",
        "What are the limitations mentioned?",
        "How does this compare to prior work?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = rlm.completion(context=context, query=query)
        print(f"Answer: {result.answer}\n")
