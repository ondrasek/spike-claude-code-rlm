"""Main RLM orchestrator class.

Implements the Recursive Language Model pattern for processing long contexts
beyond typical LLM context windows.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from .backends import LLMBackend
from .context import CompositeContext
from .prompts import get_system_prompt, get_user_prompt
from .repl import REPLEnv

_LOG_PREFIX = click.style("[RLM]", fg="yellow", bold=True)


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
            click.echo(f"{_LOG_PREFIX} {message}")

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

    @staticmethod
    def _build_context_sample(
        repl: REPLEnv,
        sample_size: int = 500,
        num_samples: int = 4,
    ) -> str:
        """Build a document sample string for the initial user prompt.

        Extracts evenly-spaced samples from the REPL context so the LLM
        can see format variations across the whole document.

        Parameters
        ----------
        repl : REPLEnv
            The REPL environment (already wraps the context).
        sample_size : int
            Characters to sample from each region.
        num_samples : int
            Number of evenly-spaced samples (minimum 2: head + tail).

        Returns
        -------
        str
            Formatted sample string.
        """
        ctx = repl.context
        size = len(ctx)

        parts = [f"Size: {size:,} characters"]

        if size <= sample_size * 2:
            parts.append(f"\nFull content preview:\n{ctx!s}")
            return "\n".join(parts)

        # Evenly-spaced sample offsets (always includes 0 and near-end)
        num_samples = max(2, num_samples)
        offsets = [size * i // num_samples for i in range(num_samples)]

        labels = ["Beginning", "~25%", "~50%", "~75%", "End"]
        for i, offset in enumerate(offsets):
            # Clamp to avoid running past end
            clamped = min(offset, size - sample_size)
            sample = ctx[clamped : clamped + sample_size]
            label = labels[i] if i < len(labels) else f"~{100 * i // num_samples}%"
            parts.append(f"\n{label} (offset {clamped:,}):\n{sample}")

        # Always include the tail
        tail_start = max(0, size - sample_size)
        tail = ctx[tail_start:]
        parts.append(f"\nEnd (offset {tail_start:,}):\n{tail}")

        return "\n".join(parts)

    def _log_context_size(self, context: str | Path | list[Path] | CompositeContext) -> None:
        """Log context size information.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The context to describe.
        """
        if isinstance(context, CompositeContext):
            self._log(
                f"Context: {len(context.files)} files, {len(context):,} bytes total (composite)"
            )
        elif isinstance(context, list):
            total = sum(p.stat().st_size for p in context)
            self._log(f"Context: {len(context)} files, {total:,} bytes total")
        elif isinstance(context, Path):
            self._log(f"Context size: {context.stat().st_size:,} bytes (memory-mapped)")
        else:
            self._log(f"Context size: {len(context):,} characters")

    def _execute_code_blocks(
        self,
        code_blocks: list[str],
        repl: REPLEnv,
    ) -> tuple[list[str], str | None]:
        """Execute code blocks and collect output.

        Parameters
        ----------
        code_blocks : list[str]
            Code blocks to execute.
        repl : REPLEnv
            REPL environment.

        Returns
        -------
        tuple[list[str], str | None]
            Tuple of (outputs, final_answer). final_answer is None if not found.
        """
        all_output: list[str] = []
        for i, code in enumerate(code_blocks):
            self._log(f"Executing code block {i + 1}/{len(code_blocks)}")
            if self.verbose:
                click.echo(click.style("Code:", fg="cyan"))
                click.echo(code)

            exec_result = repl.execute(code)

            if not exec_result.success:
                self._log(f"Execution error: {exec_result.error}")
                all_output.append(f"[Error in block {i + 1}]: {exec_result.error}")
                continue

            if exec_result.output:
                self._log(f"Output: {exec_result.output[:200]}...")
                all_output.append(exec_result.output)

            if exec_result.final_answer:
                self._log(f"Final answer received: {exec_result.final_answer[:100]}...")
                return all_output, exec_result.final_answer

        return all_output, None

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

        # Create REPL environment
        llm_query_fn = self._create_llm_query_fn(context)
        llm_query_fn.bind_stats(stats)  # type: ignore[attr-defined]
        repl = REPLEnv(
            context=context,
            llm_query_fn=llm_query_fn,
        )

        # Build conversation with injected document sample
        context_sample = self._build_context_sample(repl)
        messages = [
            {"role": "system", "content": get_system_prompt(self.compact_prompt)},
            {"role": "user", "content": get_user_prompt(query, context_sample)},
        ]

        self._log(f"Starting RLM completion for query: {query}")
        self._log_context_size(context)
        self._log(f"Model: {self.model} | Recursive model: {self.recursive_model}")
        self._log(f"Max iterations: {self.max_iterations} | Max tokens: {self.max_tokens}")
        self._log(f"Compact prompt: {self.compact_prompt}")
        system_prompt = get_system_prompt(self.compact_prompt)
        self._log(f"System prompt length: {len(system_prompt):,} chars")

        # Main iteration loop
        for iteration in range(self.max_iterations):
            stats.iterations = iteration + 1
            stats.llm_calls += 1

            iter_label = click.style(
                f"ITERATION {iteration + 1}/{self.max_iterations}",
                bold=True,
                fg="magenta",
            )
            self._log(f"\n\U0001f504 {iter_label}")
            self._log(f"Conversation messages: {len(messages)}")

            # Get LLM response
            try:
                result = self.backend.completion(messages, self.model, max_tokens=self.max_tokens)
                response = result.text
                stats.total_input_tokens += result.usage.input_tokens
                stats.total_output_tokens += result.usage.output_tokens
                self._log(
                    f"LLM response: {len(response)} chars "
                    f"(in={result.usage.input_tokens} out={result.usage.output_tokens} tokens)"
                )

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
                self._log("No python code blocks found in LLM response")
                if self.verbose:
                    preview = response[:300].replace("\n", "\n  ")
                    self._log(f"Response preview:\n  {preview}")

                # Check if LLM provided a direct answer
                if "FINAL" in response or repl.final_answer:
                    self._log("FINAL detected in response text, ending loop")
                    break

                # Prompt for code
                self._log("Prompting LLM to provide Python code")
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide Python code to explore CONTEXT.",
                    }
                )
                continue

            # Execute code blocks
            self._log(f"Found {len(code_blocks)} python code block(s)")
            all_output, final_answer = self._execute_code_blocks(code_blocks, repl)

            if final_answer:
                history.append(
                    {
                        "iteration": iteration + 1,
                        "response": response,
                        "code": code_blocks,
                        "output": "\n---\n".join(all_output),
                        "final_answer": final_answer,
                    }
                )
                return RLMResult(
                    answer=final_answer,
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
            self._log(f"Feeding {len(output_msg):,} chars of output back to LLM")
            messages.append({"role": "user", "content": f"Output:\n{output_msg}"})

        # Max iterations reached without final answer
        self._log(f"Max iterations ({self.max_iterations}) reached without FINAL() call")
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
