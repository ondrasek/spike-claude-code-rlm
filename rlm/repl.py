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

from .context import CompositeContext, LazyContext


def _materialize_context(
    context: str | Path | list[Path] | CompositeContext,
) -> tuple[str, dict[str, str] | None]:
    """Convert any supported context input to a plain ``str`` and optional files dict.

    Parameters
    ----------
    context : str | Path | list[Path] | CompositeContext
        Raw context as provided by the caller.

    Returns
    -------
    tuple[str, dict[str, str] | None]
        ``(full_text, files_dict)``.  *files_dict* is ``None`` for
        single-document inputs.
    """
    # Already a plain string — the common case.
    if type(context) is str:
        return context, None

    # Multiple files already wrapped.
    if type(context) is CompositeContext:
        files = {name: str(context.file(name)) for name in context.files}
        return str(context), files

    # list[Path] → build CompositeContext, then materialise.
    if type(context) is list:
        paths: list[Path] = context  # ty: ignore[invalid-assignment]
        composite = CompositeContext.from_paths(paths)
        files = {name: str(composite.file(name)) for name in composite.files}
        return str(composite), files

    # Single Path (or subclass like PosixPath/WindowsPath) → load, then materialise.
    if issubclass(type(context), Path):
        path: Path = context  # type: ignore[assignment]
        lazy = LazyContext(path)
        text = str(lazy)
        lazy.close()
        return text, None

    # Fallback: coerce to str.
    return str(context), None


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

# Names injected by the REPL that SHOW_VARS should hide.
_REPL_INTERNALS = frozenset(
    {
        "__builtins__",
        "CONTEXT",
        "FILES",
        "SHOW_VARS",
        "llm_query",
        "FINAL",
        "FINAL_VAR",
        "re",
        "json",
        "math",
        "collections",
        "itertools",
        "print",
    }
)


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
    - CONTEXT variable (the full document as a plain Python ``str``)
    - FILES dict (``{filename: content_str}``) when multiple files are loaded
    - SHOW_VARS() helper to list user-defined variables
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

            * ``str`` — used directly as the CONTEXT string.
            * ``Path`` — loaded via :class:`LazyContext`, then materialized.
            * ``list[Path]`` — multiple files wrapped in
              :class:`CompositeContext`, then materialized.
            * ``CompositeContext`` — materialized to ``str``.
        llm_query_fn : Callable[[str], str]
            Function to call for recursive LLM queries.
        max_output_length : int
            Maximum length of captured output.
        """
        self.context_str: str
        self.files_dict: dict[str, str] | None = None

        self.context_str, self.files_dict = _materialize_context(context)

        self.llm_query_fn = llm_query_fn
        self.max_output_length = max_output_length
        self.final_answer: str | None = None
        self.output_buffer: list[str] = []
        self._pending_final_var: str | None = None

        # Persistent namespace — survives across execute() calls.
        self._namespace: dict[str, Any] = self._build_namespace()

    def _show_vars(self) -> None:
        """Print user-defined variables in the REPL namespace."""
        user_vars = {
            k: v
            for k, v in self._namespace.items()
            if k not in _REPL_INTERNALS and not k.startswith("_")
        }
        if not user_vars:
            self._capture_print("(no user-defined variables)")
            return
        for name, value in user_vars.items():
            rep = repr(value)
            if len(rep) > 100:
                rep = rep[:97] + "..."
            self._capture_print(f"{name} = {rep}")

    def _build_namespace(self) -> dict[str, Any]:
        """Build the execution namespace with RLM helpers and safe builtins."""
        ns: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "CONTEXT": self.context_str,
            "llm_query": self._llm_query,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "re": re,
            "json": json,
            "math": math,
            "collections": collections,
            "itertools": itertools,
            "print": self._capture_print,
        }
        if self.files_dict is not None:
            ns["FILES"] = self.files_dict
        return ns

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
        self.final_answer = answer

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
            exec(code, self._namespace)  # noqa: S102  # nosec B102

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
