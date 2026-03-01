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

from .backends import LLMBackend, StructuredResponse
from .context import CompositeContext
from .prompts import (
    get_context_engineer_per_query_prompt,
    get_context_engineer_pre_loop_prompt,
    get_sub_rlm_system_prompt,
    get_system_prompt,
    get_user_prompt,
    get_verifier_system_prompt,
)
from .repl import REPLEnv

_LOG_PREFIX = click.style("[RLM]", fg="yellow", bold=True)


@dataclass
class RLMStats:
    """Statistics for an RLM completion."""

    iterations: int = 0
    llm_calls: int = 0
    sub_rlm_calls: int = 0
    context_engineer_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    structured_extractions: int = 0
    regex_extractions: int = 0


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

    Manages the interaction between the LLM, REPL environment, and sub-RLM calls
    to process contexts larger than typical LLM context windows.
    """

    def __init__(
        self,
        backend: LLMBackend,
        model: str,
        sub_rlm_model: str | None = None,
        max_iterations: int = 10,
        max_depth: int = 3,
        max_tokens: int = 4096,
        verbose: bool = False,
        compact_prompt: bool = False,
        include_context_sample: bool = True,
        timeout: float | None = None,
        max_token_budget: int | None = None,
        verify: bool = False,
        sub_rlm_backend: LLMBackend | None = None,
        verifier_backend: LLMBackend | None = None,
        verifier_model: str | None = None,
        root_system_prompt: str | None = None,
        sub_rlm_system_prompt: str | None = None,
        verifier_system_prompt: str | None = None,
        context_engineer_mode: str = "off",
        share_brief_with_root: bool = False,
        context_engineer_backend: LLMBackend | None = None,
        context_engineer_model: str | None = None,
        context_engineer_pre_loop_prompt: str | None = None,
        context_engineer_per_query_prompt: str | None = None,
    ) -> None:
        """Initialize RLM orchestrator.

        Parameters
        ----------
        backend : LLMBackend
            LLM backend to use for the root role.
        model : str
            Model identifier for root LLM.
        sub_rlm_model : str | None
            Model for sub-RLM calls (defaults to model).
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
        include_context_sample : bool
            Include document sample in initial prompt.
        timeout : float | None
            Wall-clock timeout in seconds. None means no timeout.
        max_token_budget : int | None
            Maximum total tokens (input + output). None means no limit.
        verify : bool
            Run a verification sub-call on the final answer.
        sub_rlm_backend : LLMBackend | None
            Separate backend for sub-RLM calls (defaults to backend).
        verifier_backend : LLMBackend | None
            Separate backend for verification calls (defaults to backend).
        verifier_model : str | None
            Model for verification calls (defaults to sub_rlm_model).
        root_system_prompt : str | None
            Custom system prompt for the root role.
        sub_rlm_system_prompt : str | None
            Custom system prompt for the sub-RLM role.
        verifier_system_prompt : str | None
            Custom system prompt for the verifier role.
        context_engineer_mode : str
            Context-engineer mode: "off", "pre_loop", "per_query", or "both".
        share_brief_with_root : bool
            Whether to include the document brief in the root LM's prompt.
        context_engineer_backend : LLMBackend | None
            Backend for context-engineer calls (defaults to backend).
        context_engineer_model : str | None
            Model for context-engineer calls (defaults to sub_rlm_model).
        context_engineer_pre_loop_prompt : str | None
            Custom system prompt for pre-loop analysis.
        context_engineer_per_query_prompt : str | None
            Custom system prompt for per-query enhancement.
        """
        self.backend = backend
        self.model = model
        self.sub_rlm_model = sub_rlm_model or model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.compact_prompt = compact_prompt
        self.include_context_sample = include_context_sample
        self.timeout = timeout
        self.max_token_budget = max_token_budget
        self.verify = verify
        self.sub_rlm_backend = sub_rlm_backend or backend
        self.verifier_backend = verifier_backend or backend
        self.verifier_model = verifier_model or self.sub_rlm_model
        self.root_system_prompt = root_system_prompt
        self.sub_rlm_system_prompt = sub_rlm_system_prompt
        self.verifier_system_prompt = verifier_system_prompt
        self.context_engineer_mode = context_engineer_mode
        self.share_brief_with_root = share_brief_with_root
        self.context_engineer_backend = context_engineer_backend or backend
        self.context_engineer_model = context_engineer_model or self.sub_rlm_model
        self.context_engineer_pre_loop_prompt = context_engineer_pre_loop_prompt
        self.context_engineer_per_query_prompt = context_engineer_per_query_prompt

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
    ) -> Callable[[str, str], str]:
        """Create llm_query function for REPL.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The full context.
        current_depth : int
            Current recursion depth.

        Returns
        -------
        Callable[[str, str], str]
            Function that executes LLM queries with (snippet, task) args.
        """
        stats: RLMStats | None = None  # Will be set per-completion call
        context_str: str | None = None  # Will be set per-completion call
        document_brief: str | None = None  # Will be set per-completion call

        def llm_query(snippet: str, task: str) -> str:
            """Execute a recursive LLM query.

            Parameters
            ----------
            snippet : str
                Context snippet to analyze.
            task : str
                Task/instruction for the sub-RLM.

            Returns
            -------
            str
                LLM response.
            """
            if current_depth >= self.max_depth:
                return "[ERROR: Maximum recursion depth reached]"

            if stats is not None:
                stats.sub_rlm_calls += 1
                stats.llm_calls += 1

            # Build the user prompt with optional CE enhancements
            parts: list[str] = []

            # Add document brief if available
            if document_brief:
                parts.append(f"## Document Brief\n{document_brief}")

            # Run per-query enhancement if enabled
            if self._ce_enabled_per_query() and stats is not None and context_str is not None:
                context_note = self._enhance_per_query(snippet, task, context_str, stats)
                parts.append(f"## Context Note\n{context_note}")

            parts.append(f"Context:\n{snippet}\n\nTask: {task}")
            prompt = "\n\n".join(parts)

            self._log(f"Sub-RLM call (depth={current_depth + 1}): {prompt[:100]}...")

            sub_prompt = self.sub_rlm_system_prompt or get_sub_rlm_system_prompt()
            messages = [
                {"role": "system", "content": sub_prompt},
                {"role": "user", "content": prompt},
            ]

            result = self.sub_rlm_backend.completion(
                messages, self.sub_rlm_model, max_tokens=self.max_tokens
            )

            if stats is not None:
                stats.total_input_tokens += result.usage.input_tokens
                stats.total_output_tokens += result.usage.output_tokens

            self._log(f"Sub-RLM response: {result.text[:200]}...")

            return result.text

        # Allow the caller to bind stats, context_str, and document_brief
        # after creation.
        llm_query._stats_ref = None  # type: ignore[attr-defined]

        def bind_stats(s: RLMStats) -> None:
            nonlocal stats
            stats = s

        def bind_context_str(ctx: str) -> None:
            nonlocal context_str
            context_str = ctx

        def bind_document_brief(brief: str | None) -> None:
            nonlocal document_brief
            document_brief = brief

        llm_query.bind_stats = bind_stats  # type: ignore[attr-defined]
        llm_query.bind_context_str = bind_context_str  # type: ignore[attr-defined]
        llm_query.bind_document_brief = bind_document_brief  # type: ignore[attr-defined]

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
        ctx = repl.context_str
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

    def _ce_enabled_pre_loop(self) -> bool:
        """Return True if context-engineer pre-loop analysis is enabled."""
        return self.context_engineer_mode in ("pre_loop", "both")

    def _ce_enabled_per_query(self) -> bool:
        """Return True if context-engineer per-query enhancement is enabled."""
        return self.context_engineer_mode in ("per_query", "both")

    def _run_pre_loop_analysis(
        self,
        repl: REPLEnv,
        query: str,
        stats: RLMStats,
    ) -> str:
        """Run context-engineer pre-loop analysis to produce a document brief.

        Parameters
        ----------
        repl : REPLEnv
            The REPL environment (for building context sample).
        query : str
            The user's query (provides task context).
        stats : RLMStats
            Statistics object to update.

        Returns
        -------
        str
            Document brief (200-500 words).
        """
        sample = self._build_context_sample(repl, sample_size=600, num_samples=4)
        prompt = f"User query: {query}\n\n## Document Sample\n{sample}"

        ce_prompt = self.context_engineer_pre_loop_prompt or get_context_engineer_pre_loop_prompt()
        messages = [
            {"role": "system", "content": ce_prompt},
            {"role": "user", "content": prompt},
        ]

        stats.llm_calls += 1
        stats.context_engineer_calls += 1
        self._log("Context Engineer: running pre-loop analysis")

        result = self.context_engineer_backend.completion(
            messages, self.context_engineer_model, max_tokens=self.max_tokens
        )
        stats.total_input_tokens += result.usage.input_tokens
        stats.total_output_tokens += result.usage.output_tokens

        brief = result.text.strip()
        self._log(f"Context Engineer: document brief ({len(brief)} chars)")
        if self.verbose:
            click.echo(click.style("Document Brief:", fg="cyan"))
            click.echo(brief[:500] + ("..." if len(brief) > 500 else ""))

        return brief

    def _enhance_per_query(
        self,
        snippet: str,
        task: str,
        context_str: str,
        stats: RLMStats,
    ) -> str:
        """Run context-engineer per-query enhancement to produce a context note.

        Parameters
        ----------
        snippet : str
            The snippet being passed to the sub-RLM.
        task : str
            The task description for the sub-RLM.
        context_str : str
            The full CONTEXT string for locating the snippet.
        stats : RLMStats
            Statistics object to update.

        Returns
        -------
        str
            Context note (50-150 words).
        """
        # Locate snippet in context using first 200 chars as search key
        search_key = snippet[:200]
        pos = context_str.find(search_key)

        surround_size = 500
        if pos >= 0:
            before_start = max(0, pos - surround_size)
            after_end = min(len(context_str), pos + len(snippet) + surround_size)
            before_text = context_str[before_start:pos]
            after_text = context_str[pos + len(snippet) : after_end]
            position_info = f"Snippet found at character offset {pos:,} of {len(context_str):,}"
        else:
            # Fallback: use head and tail as surrounding context
            before_text = context_str[:surround_size]
            after_text = context_str[-surround_size:]
            position_info = "Snippet location unknown (using document head/tail as context)"

        prompt = (
            f"Task for the assistant: {task}\n\n"
            f"Position: {position_info}\n\n"
            f"## Text Before Snippet\n{before_text}\n\n"
            f"## Snippet\n{snippet[:500]}{'...' if len(snippet) > 500 else ''}\n\n"
            f"## Text After Snippet\n{after_text}"
        )

        ce_prompt = (
            self.context_engineer_per_query_prompt or get_context_engineer_per_query_prompt()
        )
        messages = [
            {"role": "system", "content": ce_prompt},
            {"role": "user", "content": prompt},
        ]

        stats.llm_calls += 1
        stats.context_engineer_calls += 1
        self._log("Context Engineer: running per-query enhancement")

        result = self.context_engineer_backend.completion(
            messages, self.context_engineer_model, max_tokens=self.max_tokens
        )
        stats.total_input_tokens += result.usage.input_tokens
        stats.total_output_tokens += result.usage.output_tokens

        note = result.text.strip()
        self._log(f"Context Engineer: context note ({len(note)} chars)")

        return note

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

    def _check_guards(
        self,
        stats: RLMStats,
        history: list[dict[str, Any]],
        start_time: float,
    ) -> RLMResult | None:
        """Check timeout and token budget guards.

        Returns
        -------
        RLMResult | None
            Error result if a guard tripped, otherwise None.
        """
        import time

        if self.timeout and (time.monotonic() - start_time) > self.timeout:
            self._log("Timeout reached")
            return RLMResult(
                answer="",
                stats=stats,
                history=history,
                success=False,
                error=f"Timeout after {self.timeout}s",
            )
        total_tokens = stats.total_input_tokens + stats.total_output_tokens
        if self.max_token_budget and total_tokens > self.max_token_budget:
            self._log("Token budget exceeded")
            return RLMResult(
                answer="",
                stats=stats,
                history=history,
                success=False,
                error=f"Token budget exceeded: {total_tokens} > {self.max_token_budget}",
            )
        return None

    def _verify_answer(
        self,
        answer: str,
        context_str: str,
        stats: RLMStats,
    ) -> str:
        """Run a verification sub-call on the final answer.

        Parameters
        ----------
        answer : str
            The proposed final answer.
        context_str : str
            The full context string (truncated to first 5000 chars as evidence).
        stats : RLMStats
            Statistics object to update.

        Returns
        -------
        str
            The original answer (possibly with issues noted).
        """
        self._log("Running verification on final answer")
        evidence = context_str[:5000]
        prompt = (
            f"Proposed answer:\n{answer}\n\n"
            f"Supporting evidence (first 5000 chars of source):\n{evidence}"
        )
        v_prompt = self.verifier_system_prompt or get_verifier_system_prompt()
        messages = [
            {"role": "system", "content": v_prompt},
            {"role": "user", "content": prompt},
        ]

        stats.llm_calls += 1
        result = self.verifier_backend.completion(
            messages, self.verifier_model, max_tokens=self.max_tokens
        )
        stats.total_input_tokens += result.usage.input_tokens
        stats.total_output_tokens += result.usage.output_tokens

        verdict = result.text.strip()
        self._log(f"Verification result: {verdict[:200]}")

        if verdict.upper().startswith("ISSUES:"):
            self._log("Verification found issues")
            return f"{answer}\n\n[Verification note: {verdict}]"

        return answer

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        stats: RLMStats,
        history: list[dict[str, Any]],
    ) -> tuple[str, RLMResult | None, StructuredResponse | None]:
        """Call the root LLM and update stats.

        When the backend supports structured output, calls
        ``structured_completion()`` instead of ``completion()`` and returns the
        parsed ``StructuredResponse`` (or ``None`` when parsing fails).

        Returns
        -------
        tuple[str, RLMResult | None, StructuredResponse | None]
            ``(response_text, error_result, structured)``.  ``error_result`` is
            ``None`` on success; ``structured`` is ``None`` when the backend
            does not support structured output or parsing failed.
        """
        try:
            if self.backend.supports_structured_output:
                result = self.backend.structured_completion(
                    messages, self.model, max_tokens=self.max_tokens
                )
            else:
                result = self.backend.completion(messages, self.model, max_tokens=self.max_tokens)
        except Exception as e:
            error_msg = f"LLM call failed: {e!s}"
            self._log(f"ERROR: {error_msg}")
            return (
                "",
                RLMResult(
                    answer="",
                    stats=stats,
                    history=history,
                    success=False,
                    error=error_msg,
                ),
                None,
            )

        stats.total_input_tokens += result.usage.input_tokens
        stats.total_output_tokens += result.usage.output_tokens
        self._log(
            f"LLM response: {len(result.text)} chars "
            f"(in={result.usage.input_tokens} out={result.usage.output_tokens} tokens)"
        )
        return result.text, None, result.structured

    def _finalize_answer(
        self,
        answer: str,
        repl: REPLEnv,
        stats: RLMStats,
    ) -> str:
        """Optionally verify and return the final answer.

        Returns
        -------
        str
            The (possibly verified) final answer.
        """
        if self.verify:
            return self._verify_answer(answer, repl.context_str, stats)
        return answer

    def _extract_code_from_response(
        self,
        response: str,
        structured: StructuredResponse | None,
        stats: RLMStats,
        repl: REPLEnv,
        iteration: int,
        history: list[dict[str, Any]],
    ) -> tuple[list[str], RLMResult | None]:
        """Extract code blocks from structured output or regex, handling direct finals.

        Parameters
        ----------
        response : str
            Raw LLM response text.
        structured : StructuredResponse | None
            Parsed structured output (``None`` when unavailable).
        stats : RLMStats
            Statistics object to update.
        repl : REPLEnv
            REPL environment (for finalization).
        iteration : int
            Current iteration number (1-based).
        history : list[dict[str, Any]]
            History list to append to on direct final.

        Returns
        -------
        tuple[list[str], RLMResult | None]
            ``(code_blocks, direct_result)``.  When ``direct_result`` is not
            ``None`` the caller should return it immediately (structured final).
        """
        # Path A: structured + is_final → return answer directly
        if structured is not None and structured.is_final and structured.final_answer is not None:
            stats.structured_extractions += 1
            if structured.code is not None:
                self._log("Structured is_final with code set; ignoring code")
            structured_answer = self._finalize_answer(structured.final_answer, repl, stats)
            history.append(
                {
                    "iteration": iteration,
                    "response": response,
                    "code": [],
                    "output": "",
                    "final_answer": structured_answer,
                }
            )
            return [], RLMResult(
                answer=structured_answer,
                stats=stats,
                history=history,
                success=True,
            )

        # Path B: structured + code → use structured code
        if structured is not None and structured.code is not None:
            stats.structured_extractions += 1
            self._log("Using structured code extraction")
            return [structured.code], None

        # Path C: regex fallback
        code_blocks = self._extract_code_blocks(response)
        if code_blocks:
            stats.regex_extractions += 1
        return code_blocks, None

    def _handle_no_code_blocks(
        self,
        response: str,
        repl: REPLEnv,
        messages: list[dict[str, str]],
    ) -> bool:
        """Handle an LLM response that contains no Python code blocks.

        Returns
        -------
        bool
            True if the loop should break (FINAL detected), False to continue.
        """
        self._log("No python code blocks found in LLM response")
        if self.verbose:
            preview = response[:300].replace("\n", "\n  ")
            self._log(f"Response preview:\n  {preview}")

        if "FINAL" in response or repl.final_answer:
            self._log("FINAL detected in response text, ending loop")
            return True

        self._log("Prompting LLM to provide Python code")
        messages.append(
            {"role": "user", "content": "Please provide Python code to explore CONTEXT."}
        )
        return False

    def _setup_completion(
        self,
        context: str | Path | list[Path] | CompositeContext,
        query: str,
        stats: RLMStats,
    ) -> tuple[REPLEnv, list[dict[str, str]]]:
        """Set up REPL, context-engineer, and initial messages for completion.

        Returns
        -------
        tuple[REPLEnv, list[dict[str, str]]]
            The REPL environment and the initial message list.
        """
        llm_query_fn = self._create_llm_query_fn(context)
        llm_query_fn.bind_stats(stats)  # type: ignore[attr-defined]
        repl = REPLEnv(context=context, llm_query_fn=llm_query_fn)

        # Bind context string for per-query CE enhancement
        llm_query_fn.bind_context_str(repl.context_str)  # type: ignore[attr-defined]

        # Run context-engineer pre-loop analysis if enabled
        brief: str | None = None
        if self._ce_enabled_pre_loop():
            brief = self._run_pre_loop_analysis(repl, query, stats)
            llm_query_fn.bind_document_brief(brief)  # type: ignore[attr-defined]

        # Build conversation with injected document sample
        context_sample = self._build_context_sample(repl) if self.include_context_sample else ""
        root_prompt = self.root_system_prompt or get_system_prompt(self.compact_prompt)
        user_prompt = get_user_prompt(query, context_sample)

        if brief and self.share_brief_with_root:
            user_prompt = f"## Document Brief\n{brief}\n\n{user_prompt}"

        messages = [
            {"role": "system", "content": root_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self._log(f"Starting RLM completion for query: {query}")
        self._log_context_size(context)
        self._log(f"Model: {self.model} | Sub-RLM model: {self.sub_rlm_model}")
        self._log(f"Max iterations: {self.max_iterations} | Max tokens: {self.max_tokens}")
        self._log(f"Compact prompt: {self.compact_prompt}")
        self._log(f"Context engineer mode: {self.context_engineer_mode}")
        self._log(f"System prompt length: {len(root_prompt):,} chars")

        return repl, messages

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
        import time

        stats = RLMStats()
        self._last_stats = stats
        history: list[dict[str, Any]] = []

        repl, messages = self._setup_completion(context, query, stats)

        start_time = time.monotonic()

        # Main iteration loop
        for iteration in range(self.max_iterations):
            guard_result = self._check_guards(stats, history, start_time)
            if guard_result is not None:
                return guard_result

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
            response, error_result, structured = self._call_llm(messages, stats, history)
            if error_result is not None:
                return error_result

            # Add to conversation history
            messages.append({"role": "assistant", "content": response})

            # Extract code blocks (structured output or regex fallback)
            code_blocks, direct_result = self._extract_code_from_response(
                response, structured, stats, repl, iteration + 1, history
            )
            if direct_result is not None:
                return direct_result

            if not code_blocks:
                if self._handle_no_code_blocks(response, repl, messages):
                    break
                continue

            # Execute code blocks
            self._log(f"Found {len(code_blocks)} python code block(s)")
            all_output, final_answer = self._execute_code_blocks(code_blocks, repl)

            if final_answer:
                final_answer = self._finalize_answer(final_answer, repl, stats)
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
                "sub_rlm_calls": 0,
                "context_engineer_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "structured_extractions": 0,
                "regex_extractions": 0,
            }
        return {
            "iterations": self._last_stats.iterations,
            "llm_calls": self._last_stats.llm_calls,
            "sub_rlm_calls": self._last_stats.sub_rlm_calls,
            "context_engineer_calls": self._last_stats.context_engineer_calls,
            "total_input_tokens": self._last_stats.total_input_tokens,
            "total_output_tokens": self._last_stats.total_output_tokens,
            "structured_extractions": self._last_stats.structured_extractions,
            "regex_extractions": self._last_stats.regex_extractions,
        }
