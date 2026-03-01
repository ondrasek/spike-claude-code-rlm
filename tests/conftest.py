"""Shared fixtures for RLM test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from rlm.backends import (
    CallbackBackend,
    CompletionResult,  # noqa: F811 — used by StructuredBackend
)
from rlm.context import CompositeContext, LazyContext, StringContext
from rlm.repl import REPLEnv


def pytest_configure(config: pytest.Config) -> None:
    """Load .env before test collection so API keys are available for skip checks."""
    load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Sample text content
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """\
Chapter 1: Introduction
This is the introduction to the document.
It covers the basics of the topic.

Chapter 2: Methods
The methods section describes the approach.
Multiple techniques were used in the analysis.

Chapter 3: Results
The results show significant improvements.
Data was collected over a period of six months.

Chapter 4: Conclusion
In conclusion, the study demonstrates clear benefits.
Future work will focus on scalability.
"""

SAMPLE_TEXT_SMALL = "Hello, world!"

SAMPLE_TEXT_MULTILINE = "line1\nline2\nline3\n"


# ---------------------------------------------------------------------------
# Temp file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file with SAMPLE_TEXT."""
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
def tmp_empty_file(tmp_path: Path) -> Path:
    """Create an empty temporary file."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_multifile_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple text files."""
    (tmp_path / "file_a.txt").write_text("Contents of file A.\nSecond line A.", encoding="utf-8")
    (tmp_path / "file_b.txt").write_text("Contents of file B.\nSecond line B.", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file_c.txt").write_text("Contents of file C in subdir.", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Context fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def string_context() -> StringContext:
    """StringContext wrapping SAMPLE_TEXT."""
    return StringContext(SAMPLE_TEXT)


@pytest.fixture()
def lazy_context(tmp_text_file: Path) -> LazyContext:
    """LazyContext backed by a temp file."""
    ctx = LazyContext(tmp_text_file)
    yield ctx  # type: ignore[misc]
    ctx.close()


@pytest.fixture()
def empty_lazy_context(tmp_empty_file: Path) -> LazyContext:
    """LazyContext backed by an empty file."""
    ctx = LazyContext(tmp_empty_file)
    yield ctx  # type: ignore[misc]
    ctx.close()


@pytest.fixture()
def composite_context(tmp_multifile_dir: Path) -> CompositeContext:
    """CompositeContext from tmp_multifile_dir."""
    ctx = CompositeContext.from_directory(tmp_multifile_dir)
    yield ctx  # type: ignore[misc]
    ctx.close()


# ---------------------------------------------------------------------------
# Backend / mock fixtures
# ---------------------------------------------------------------------------


def make_echo_callback() -> CallbackBackend:
    """Backend that echoes the last user message content."""

    def _echo(messages: list[dict[str, str]], model: str) -> str:
        return messages[-1]["content"] if messages else ""

    return CallbackBackend(_echo)


def make_deterministic_callback(responses: list[str]) -> CallbackBackend:
    """Backend that returns responses from a list in order, cycling."""
    idx = {"i": 0}

    def _cb(messages: list[dict[str, str]], model: str) -> str:
        resp = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return resp

    return CallbackBackend(_cb)


def make_final_in_two_iterations_callback() -> CallbackBackend:
    """Backend that explores on iteration 1 and calls FINAL on iteration 2."""
    responses = [
        # Iteration 1 — explore context
        '```python\nprint(f"Size: {len(CONTEXT)}")\n```',
        # Iteration 2 — produce final answer
        '```python\nFINAL("The answer is 42")\n```',
    ]
    return make_deterministic_callback(responses)


@pytest.fixture()
def echo_backend() -> CallbackBackend:
    """CallbackBackend that echoes the last message."""
    return make_echo_callback()


@pytest.fixture()
def final_two_iter_backend() -> CallbackBackend:
    """CallbackBackend that produces FINAL on the 2nd iteration."""
    return make_final_in_two_iterations_callback()


# ---------------------------------------------------------------------------
# REPL fixtures
# ---------------------------------------------------------------------------


def noop_llm_query(snippet: str, task: str) -> str:
    """No-op LLM query function."""
    return f"[mock response to: {task[:50]}]"


@pytest.fixture()
def repl_env() -> REPLEnv:
    """REPLEnv with SAMPLE_TEXT and a no-op llm_query."""
    return REPLEnv(context=SAMPLE_TEXT, llm_query_fn=noop_llm_query)


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------


class StructuredBackend(CallbackBackend):
    """Backend that supports structured output for testing.

    Wraps a list of ``CompletionResult`` objects (with ``.structured`` set)
    and cycles through them on successive calls.
    """

    def __init__(self, results: list[CompletionResult]) -> None:
        self._results = results
        self._idx = 0

        # Provide a dummy callback for the parent class
        super().__init__(lambda msgs, model: "")

    @property
    def supports_structured_output(self) -> bool:  # type: ignore[override]
        return True

    def structured_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: object,
    ) -> CompletionResult:
        result = self._results[self._idx % len(self._results)]
        self._idx += 1
        return result

    def completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: object,
    ) -> CompletionResult:
        # Shouldn't be called when supports_structured_output is True,
        # but delegate to structured_completion as a safety net.
        return self.structured_completion(messages, model, **kwargs)


def make_structured_callback(
    responses: list[CompletionResult],
) -> StructuredBackend:
    """Create a ``StructuredBackend`` from a list of pre-built results."""
    return StructuredBackend(responses)
