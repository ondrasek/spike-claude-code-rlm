"""Main RLM orchestrator class.

Implements the Recursive Language Model pattern for processing long contexts
beyond typical LLM context windows.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .backends import LLMBackend
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
        verbose: bool = False,
        compact_prompt: bool = False,
    ) -> None:
        """Initialize RLM orchestrator.

        Args:
            backend: LLM backend to use
            model: Model identifier for root LLM
            recursive_model: Model for llm_query calls (defaults to model)
            max_iterations: Maximum REPL iterations
            max_depth: Maximum recursion depth for llm_query
            verbose: Print debug output
            compact_prompt: Use shorter system prompt
        """
        self.backend = backend
        self.model = model
        self.recursive_model = recursive_model or model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.verbose = verbose
        self.compact_prompt = compact_prompt
        self.stats = RLMStats()

    def _log(self, message: str) -> None:
        """Log a message if verbose is enabled.

        Args:
            message: Message to log
        """
        if self.verbose:
            print(f"[RLM] {message}")

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract Python code blocks from LLM response.

        Args:
            text: LLM response text

        Returns:
            List of code blocks
        """
        # Match ```python ... ``` blocks
        pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _create_llm_query_fn(
        self, context: str, current_depth: int = 0
    ) -> callable[[str], str]:
        """Create llm_query function for REPL.

        Args:
            context: The full context
            current_depth: Current recursion depth

        Returns:
            Function that executes LLM queries
        """

        def llm_query(prompt: str) -> str:
            """Execute a recursive LLM query.

            Args:
                prompt: Query prompt

            Returns:
                LLM response
            """
            if current_depth >= self.max_depth:
                return "[ERROR: Maximum recursion depth reached]"

            self.stats.recursive_calls += 1
            self.stats.llm_calls += 1

            self._log(f"Recursive call (depth={current_depth + 1}): {prompt[:100]}...")

            # For recursive calls, we use a simpler prompt
            messages = [
                {"role": "user", "content": prompt},
            ]

            response = self.backend.completion(messages, self.recursive_model)

            self._log(f"Recursive response: {response[:200]}...")

            return response

        return llm_query

    def completion(self, context: str, query: str) -> RLMResult:
        """Execute RLM completion.

        Args:
            context: The full document/context to process
            query: User's query

        Returns:
            RLMResult with answer and statistics
        """
        self.stats = RLMStats()
        history: list[dict[str, Any]] = []

        # Build conversation
        messages = [
            {"role": "system", "content": get_system_prompt(self.compact_prompt)},
            {"role": "user", "content": get_user_prompt(query)},
        ]

        # Create REPL environment
        repl = REPLEnv(
            context=context,
            llm_query_fn=self._create_llm_query_fn(context),
        )

        self._log(f"Starting RLM completion for query: {query}")
        self._log(f"Context size: {len(context):,} characters")

        # Main iteration loop
        for iteration in range(self.max_iterations):
            self.stats.iterations = iteration + 1
            self.stats.llm_calls += 1

            self._log(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # Get LLM response
            try:
                response = self.backend.completion(messages, self.model)
                self._log(f"LLM response length: {len(response)}")

            except Exception as e:
                error_msg = f"LLM call failed: {e!s}"
                self._log(f"ERROR: {error_msg}")
                return RLMResult(
                    answer="",
                    stats=self.stats,
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

                result = repl.execute(code)

                if not result.success:
                    self._log(f"Execution error: {result.error}")
                    all_output.append(f"[Error in block {i + 1}]: {result.error}")
                else:
                    if result.output:
                        self._log(f"Output: {result.output[:200]}...")
                        all_output.append(result.output)

                    if result.final_answer:
                        self._log(f"Final answer received: {result.final_answer[:100]}...")
                        # Record this iteration
                        history.append(
                            {
                                "iteration": iteration + 1,
                                "response": response,
                                "code": code_blocks,
                                "output": "\n---\n".join(all_output),
                                "final_answer": result.final_answer,
                            }
                        )

                        return RLMResult(
                            answer=result.final_answer,
                            stats=self.stats,
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
            stats=self.stats,
            history=history,
            success=False,
            error="Maximum iterations reached without final answer",
        )

    async def acompletion(self, context: str, query: str) -> RLMResult:
        """Async version of completion.

        Args:
            context: The full document/context
            query: User's query

        Returns:
            RLMResult with answer and statistics
        """
        # For simplicity, delegate to sync version
        # Full async implementation would require async REPL execution
        return self.completion(context, query)

    def cost_summary(self) -> dict[str, Any]:
        """Get usage statistics summary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "iterations": self.stats.iterations,
            "llm_calls": self.stats.llm_calls,
            "recursive_calls": self.stats.recursive_calls,
            "total_input_tokens": self.stats.total_input_tokens,
            "total_output_tokens": self.stats.total_output_tokens,
        }
