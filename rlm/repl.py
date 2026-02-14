"""REPL environment for executing LLM-generated Python code.

This module provides a Python execution environment where the LLM can explore
the CONTEXT variable through code. Container-level isolation is expected to be
provided by the runtime (e.g. a rootless container).
"""

import collections
import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class REPLResult:
    """Result from REPL execution."""

    output: str
    final_answer: str | None = None
    error: str | None = None
    success: bool = True


class REPLEnv:
    """REPL environment for RLM code execution.

    Provides:
    - CONTEXT variable (the full document)
    - llm_query() function for recursive LLM calls
    - FINAL() and FINAL_VAR() for returning results
    - Pre-imported modules: re, json, math, collections, itertools

    Isolation is delegated to the container runtime.
    """

    def __init__(
        self,
        context: str,
        llm_query_fn: Callable[[str], str],
        max_output_length: int = 10000,
    ) -> None:
        """Initialize REPL environment.

        Args:
            context: The full document/context string
            llm_query_fn: Function to call for recursive LLM queries
            max_output_length: Maximum length of captured output
        """
        self.context = context
        self.llm_query_fn = llm_query_fn
        self.max_output_length = max_output_length
        self.final_answer: str | None = None
        self.output_buffer: list[str] = []

    def _capture_print(self, *args: Any, **kwargs: Any) -> None:
        """Capture print output to buffer."""
        output = " ".join(str(arg) for arg in args)
        if kwargs.get("sep"):
            output = str(kwargs["sep"]).join(str(arg) for arg in args)
        self.output_buffer.append(output)

    def _llm_query(self, prompt: str) -> str:
        """Wrapper for LLM query function.

        Args:
            prompt: Query to send to the LLM

        Returns:
            LLM response string
        """
        return self.llm_query_fn(prompt)

    def _final(self, answer: str) -> None:
        """Set the final answer.

        Args:
            answer: Final answer string
        """
        self.final_answer = str(answer)

    def _final_var(self, var_name: str) -> None:
        """Set a variable as the final answer.

        Args:
            var_name: Name of variable to use as final answer
        """
        # Will be handled by reading from namespace
        pass

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with output, final answer, and any errors
        """
        # Reset output buffer
        self.output_buffer = []

        # Create namespace with RLM helpers and common stdlib modules
        namespace: dict[str, Any] = {
            "CONTEXT": self.context,
            "llm_query": self._llm_query,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "re": re,
            "json": json,
            "math": math,
            "collections": collections,
            "itertools": itertools,
            "print": self._capture_print,
        }

        # Execute code
        try:
            exec(code, namespace)  # noqa: S102

            # Check if FINAL_VAR was called
            if self.final_answer is None:
                # Look for variables that might have been set as final
                for key, value in namespace.items():
                    if key.startswith("final_") and not key.startswith("_"):
                        self.final_answer = str(value)
                        break

            # Collect output
            output = "\n".join(self.output_buffer)
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n[...output truncated]"

            return REPLResult(
                output=output,
                final_answer=self.final_answer,
                success=True,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}"
            return REPLResult(
                output="\n".join(self.output_buffer),
                error=error_msg,
                success=False,
            )
