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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import CompositeContext, LazyContext, StringContext

# Restricted set of builtins safe for the REPL sandbox.
# This prevents LLM-generated code from importing arbitrary modules,
# accessing the filesystem, or executing shell commands.
_SAFE_BUILTINS: dict[str, Any] = {
    # Types and constructors
    "True": True,
    "False": False,
    "None": None,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "bytes": bytes,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "complex": complex,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "object": object,
    "type": type,
    "slice": slice,
    "range": range,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # Iteration and comprehension
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "iter": iter,
    "next": next,
    # Numeric and math
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    # String and representation
    "repr": repr,
    "ascii": ascii,
    "chr": chr,
    "ord": ord,
    "format": format,
    "hash": hash,
    # Collections and sorting
    "len": len,
    "sorted": sorted,
    "any": any,
    "all": all,
    # Type checking
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "id": id,
    "dir": dir,
    "vars": vars,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    # Exceptions (needed for try/except)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "OverflowError": OverflowError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "UnicodeError": UnicodeError,
    "UnicodeDecodeError": UnicodeDecodeError,
    "UnicodeEncodeError": UnicodeEncodeError,
    "NotImplementedError": NotImplementedError,
}


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

    The namespace is preserved across ``execute()`` calls within the same
    ``REPLEnv`` instance, giving true REPL semantics where variables defined
    in one code block are available in subsequent blocks.
    """

    def __init__(
        self,
        context: str | Path | list[Path] | CompositeContext,
        llm_query_fn: Callable[[str], str],
        max_output_length: int = 10000,
    ) -> None:
        """Initialize REPL environment.

        Parameters
        ----------
        context : str | Path | list[Path] | CompositeContext
            The document/context.

            * ``str`` — wrapped in :class:`StringContext`.
            * ``Path`` — memory-mapped via :class:`LazyContext`.
            * ``list[Path]`` — multiple files wrapped in
              :class:`CompositeContext`.
            * ``CompositeContext`` — used as-is.
        llm_query_fn : Callable[[str], str]
            Function to call for recursive LLM queries.
        max_output_length : int
            Maximum length of captured output.
        """
        if isinstance(context, CompositeContext):
            self.context: LazyContext | StringContext | CompositeContext = context
        elif isinstance(context, list):
            self.context = CompositeContext.from_paths(context)
        elif isinstance(context, Path):
            self.context = LazyContext(context)
        else:
            self.context = StringContext(context)
        self.llm_query_fn = llm_query_fn
        self.max_output_length = max_output_length
        self.final_answer: str | None = None
        self.output_buffer: list[str] = []
        self._pending_final_var: str | None = None

        # Persistent namespace — survives across execute() calls.
        self._namespace: dict[str, Any] = self._build_namespace()

    def _build_namespace(self) -> dict[str, Any]:
        """Build the execution namespace with RLM helpers and safe builtins."""
        return {
            "__builtins__": dict(_SAFE_BUILTINS),
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

    def _capture_print(self, *args: Any, **kwargs: Any) -> None:
        """Capture print output to buffer.

        Supports ``sep`` and ``end`` keyword arguments matching the built-in
        ``print()`` signature.  The ``file`` and ``flush`` kwargs are accepted
        but ignored (output always goes to the internal buffer).
        """
        sep_val = kwargs.get("sep")
        end_val = kwargs.get("end")
        sep: str = str(sep_val) if sep_val is not None else " "
        end: str = str(end_val) if end_val is not None else "\n"
        output = sep.join(str(arg) for arg in args) + end
        self.output_buffer.append(output)

    def _llm_query(self, prompt: str) -> str:
        """Wrapper for LLM query function.

        Parameters
        ----------
        prompt : str
            Query to send to the LLM.

        Returns
        -------
        str
            LLM response string.
        """
        return self.llm_query_fn(prompt)

    def _final(self, answer: str) -> None:
        """Set the final answer.

        Parameters
        ----------
        answer : str
            Final answer string.
        """
        self.final_answer = str(answer)

    def _final_var(self, var_name: str) -> None:
        """Mark a variable as the final answer.

        The variable will be looked up in the execution namespace after the
        current code block finishes executing.

        Parameters
        ----------
        var_name : str
            Name of the variable whose value should be used as the final answer.
        """
        self._pending_final_var = var_name

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the REPL environment.

        The namespace persists across calls, so variables set in one code
        block are available in subsequent blocks.

        Parameters
        ----------
        code : str
            Python code to execute.

        Returns
        -------
        REPLResult
            Result with output, final answer, and any errors.
        """
        # Reset per-execution state (but NOT self.final_answer — that is only
        # set by explicit FINAL() / FINAL_VAR() calls within the code).
        self.output_buffer = []
        self._pending_final_var = None

        try:
            exec(code, self._namespace)  # noqa: S102

            # Resolve FINAL_VAR if it was called during execution.
            if self.final_answer is None and self._pending_final_var is not None:
                var_name = self._pending_final_var
                if var_name in self._namespace:
                    self.final_answer = str(self._namespace[var_name])
                else:
                    return REPLResult(
                        output="".join(self.output_buffer),
                        error=f"FINAL_VAR: variable '{var_name}' not found in namespace",
                        success=False,
                    )

            # Fall back to variables prefixed with ``final_`` (legacy behaviour).
            if self.final_answer is None:
                for key, value in self._namespace.items():
                    if key.startswith("final_") and not key.startswith("__"):
                        self.final_answer = str(value)
                        break

            # Collect output — join without extra separator since _capture_print
            # already appends the ``end`` string per call.
            output = "".join(self.output_buffer)
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
                output="".join(self.output_buffer),
                error=error_msg,
                success=False,
            )
