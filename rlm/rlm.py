"""Main RLM orchestrator class.

Implements the Recursive Language Model pattern for processing long contexts
beyond typical LLM context windows.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import LLMBackend
from .context import CompositeContext
from .prompts import get_system_prompt, get_user_prompt
from .repl import REPLEnv


@dataclass
class RLMStats:
    """Statistics for an RLM completion."""

    iterations: int = 0
    llm_calls: int = 0
    recursive_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class RLMResult:
    """Result from RLM completion."""

    answer: str
    stats: RLMStats
    history: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class RLM:
    """Recursive Language Model orchestrator.

    Manages the interaction between the LLM, REPL environment, and recursive calls
    to process contexts larger than typical LLM context windows.
    """

    def __init__(
        self,
        backend: LLMBackend,
        model: str = "claude-sonnet-4-20250514",
        recursive_model: str | None = None,
        max_iterations: int = 10,
        max_depth: int = 3,
        max_tokens: int = 4096,
        verbose: bool = False,
        compact_prompt: bool = False,
    ) -> None:
        """Initialize RLM orchestrator.

        Parameters
        ----------
        backend : LLMBackend
            LLM backend to use.
        model : str
            Model identifier for root LLM.
        recursive_model : str | None
            Model for llm_query calls (defaults to model).
        max_iterations : int
            Maximum REPL iterations.
        max_depth : int
            Maximum recursion depth for llm_query.
        max_tokens : int
            Maximum tokens per LLM response.
        verbose : bool
            Print debug output.
        compact_prompt : bool
            Use shorter system prompt.
        """
        self.backend = backend
        self.model = model
        self.recursive_model = recursive_model or model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.compact_prompt = compact_prompt

    def _log(self, message: str) -> None:
        """Log a message if verbose is enabled.

        Parameters
        ----------
        message : str
            Message to log.
        """
        if self.verbose:
            print(f"[RLM] {message}")

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract Python code blocks from LLM response.

        Only matches fenced code blocks explicitly tagged as ``python``.
        Untagged or differently-tagged blocks (e.g. ``bash``, ``json``) are
        ignored to avoid accidentally executing non-Python code.

        Parameters
        ----------
        text : str
            LLM response text.

        Returns
        -------
        list[str]
            List of Python code strings.
        """
        pattern = r"```python\n(.*?)```"
        return re.findall(pattern, text, re.DOTALL)

    def _create_llm_query_fn(
        self,
        context: str | Path | list[Path] | CompositeContext,
        current_depth: int = 0,
    ) -> Callable[[str], str]:
        """Create llm_query function for REPL.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The full context.
        current_depth : int
            Current recursion depth.

        Returns
        -------
        Callable[[str], str]
            Function that executes LLM queries.
        """
        stats = None  # Will be set per-completion call

        def llm_query(prompt: str) -> str:
            """Execute a recursive LLM query.

            Parameters
            ----------
            prompt : str
                Query prompt.

            Returns
            -------
            str
                LLM response.
            """
            if current_depth >= self.max_depth:
                return "[ERROR: Maximum recursion depth reached]"

            if stats is not None:
                stats.recursive_calls += 1
                stats.llm_calls += 1

            self._log(f"Recursive call (depth={current_depth + 1}): {prompt[:100]}...")

            messages = [
                {"role": "user", "content": prompt},
            ]

            result = self.backend.completion(
                messages, self.recursive_model, max_tokens=self.max_tokens
            )

            if stats is not None:
                stats.total_input_tokens += result.usage.input_tokens
                stats.total_output_tokens += result.usage.output_tokens

            self._log(f"Recursive response: {result.text[:200]}...")

            return result.text

        # Allow the caller to bind stats after creation.
        llm_query._stats_ref = None  # type: ignore[attr-defined]

        def bind_stats(s: RLMStats) -> None:
            nonlocal stats
            stats = s

        llm_query.bind_stats = bind_stats  # type: ignore[attr-defined]

        return llm_query

    def completion(
        self, context: str | Path | list[Path] | CompositeContext, query: str
    ) -> RLMResult:
        """Execute RLM completion.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The document/context to process.

            * ``str`` — in-memory string.
            * ``Path`` — single file, memory-mapped.
            * ``list[Path]`` — multiple files, each memory-mapped.
            * ``CompositeContext`` — pre-built composite.
        query : str
            User's query.

        Returns
        -------
        RLMResult
            Result with answer and statistics.
        """
        stats = RLMStats()
        self._last_stats = stats
        history: list[dict[str, Any]] = []

        # Build conversation
        messages = [
            {"role": "system", "content": get_system_prompt(self.compact_prompt)},
            {"role": "user", "content": get_user_prompt(query)},
        ]

        # Create REPL environment
        llm_query_fn = self._create_llm_query_fn(context)
        llm_query_fn.bind_stats(stats)  # type: ignore[attr-defined]
        repl = REPLEnv(
            context=context,
            llm_query_fn=llm_query_fn,
        )

        self._log(f"Starting RLM completion for query: {query}")

        # Determine displayable context size.
        if isinstance(context, CompositeContext):
            self._log(
                f"Context: {len(context.files)} files, "
                f"{len(context):,} bytes total (composite)"
            )
        elif isinstance(context, list):
            total = sum(p.stat().st_size for p in context)
            self._log(f"Context: {len(context)} files, {total:,} bytes total")
        elif isinstance(context, Path):
            self._log(f"Context size: {context.stat().st_size:,} bytes (memory-mapped)")
        else:
            self._log(f"Context size: {len(context):,} characters")

        # Main iteration loop
        for iteration in range(self.max_iterations):
            stats.iterations = iteration + 1
            stats.llm_calls += 1

            self._log(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # Get LLM response
            try:
                result = self.backend.completion(
                    messages, self.model, max_tokens=self.max_tokens
                )
                response = result.text
                stats.total_input_tokens += result.usage.input_tokens
                stats.total_output_tokens += result.usage.output_tokens
                self._log(f"LLM response length: {len(response)}")

            except Exception as e:
                error_msg = f"LLM call failed: {e!s}"
                self._log(f"ERROR: {error_msg}")
                return RLMResult(
                    answer="",
                    stats=stats,
                    history=history,
                    success=False,
                    error=error_msg,
                )

            # Add to conversation history
            messages.append({"role": "assistant", "content": response})

            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response)

            if not code_blocks:
                self._log("No code blocks found in response")

                # Check if LLM provided a direct answer
                if "FINAL" in response or repl.final_answer:
                    break

                # Prompt for code
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide Python code to explore CONTEXT.",
                    }
                )
                continue

            # Execute each code block
            all_output = []
            for i, code in enumerate(code_blocks):
                self._log(f"Executing code block {i + 1}/{len(code_blocks)}")
                if self.verbose:
                    print(f"Code:\n{code}\n")

                exec_result = repl.execute(code)

                if not exec_result.success:
                    self._log(f"Execution error: {exec_result.error}")
                    all_output.append(f"[Error in block {i + 1}]: {exec_result.error}")
                else:
                    if exec_result.output:
                        self._log(f"Output: {exec_result.output[:200]}...")
                        all_output.append(exec_result.output)

                    if exec_result.final_answer:
                        self._log(
                            f"Final answer received: {exec_result.final_answer[:100]}..."
                        )
                        # Record this iteration
                        history.append(
                            {
                                "iteration": iteration + 1,
                                "response": response,
                                "code": code_blocks,
                                "output": "\n---\n".join(all_output),
                                "final_answer": exec_result.final_answer,
                            }
                        )

                        return RLMResult(
                            answer=exec_result.final_answer,
                            stats=stats,
                            history=history,
                            success=True,
                        )

            # Record iteration
            history.append(
                {
                    "iteration": iteration + 1,
                    "response": response,
                    "code": code_blocks,
                    "output": "\n---\n".join(all_output),
                }
            )

            # Provide output back to LLM
            output_msg = "\n---\n".join(all_output) if all_output else "(no output)"
            messages.append({"role": "user", "content": f"Output:\n{output_msg}"})

        # Max iterations reached without final answer
        self._log("Max iterations reached without final answer")
        return RLMResult(
            answer="",
            stats=stats,
            history=history,
            success=False,
            error="Maximum iterations reached without final answer",
        )

    async def acompletion(
        self, context: str | Path | list[Path] | CompositeContext, query: str
    ) -> RLMResult:
        """Async version of completion.

        Note: This currently delegates to the synchronous ``completion()``
        method and will block the event loop.  A fully async implementation
        would require an async REPL execution model.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The document/context.
        query : str
            User's query.

        Returns
        -------
        RLMResult
            Result with answer and statistics.
        """
        return self.completion(context, query)

    def cost_summary(self) -> dict[str, Any]:
        """Get usage statistics summary.

        Returns
        -------
        dict[str, Any]
            Dictionary with usage statistics.
        """
        # Stats are only available after calling completion().
        # Return zeros if no completion has been run.
        if not hasattr(self, "_last_stats"):
            return {
                "iterations": 0,
                "llm_calls": 0,
                "recursive_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        return {
            "iterations": self._last_stats.iterations,
            "llm_calls": self._last_stats.llm_calls,
            "recursive_calls": self._last_stats.recursive_calls,
            "total_input_tokens": self._last_stats.total_input_tokens,
            "total_output_tokens": self._last_stats.total_output_tokens,
        }
