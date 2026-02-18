"""Unit tests for the RLM orchestrator (rlm/rlm.py)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rlm.backends import CallbackBackend
from rlm.context import CompositeContext
from rlm.rlm import RLM, RLMResult, RLMStats

from .conftest import (
    SAMPLE_TEXT,
    make_deterministic_callback,
    make_final_in_two_iterations_callback,
    noop_llm_query,
)

# =====================================================================
# RLMStats dataclass
# =====================================================================


class TestRLMStats:
    """Tests for the RLMStats dataclass."""

    def test_default_values(self) -> None:
        stats = RLMStats()
        assert stats.iterations == 0
        assert stats.llm_calls == 0
        assert stats.recursive_calls == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0

    def test_custom_initialization(self) -> None:
        stats = RLMStats(
            iterations=5,
            llm_calls=10,
            recursive_calls=2,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert stats.iterations == 5
        assert stats.llm_calls == 10
        assert stats.recursive_calls == 2
        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 500

    def test_field_mutation(self) -> None:
        stats = RLMStats()
        stats.iterations = 3
        stats.llm_calls = 7
        assert stats.iterations == 3
        assert stats.llm_calls == 7


# =====================================================================
# RLMResult dataclass
# =====================================================================


class TestRLMResult:
    """Tests for the RLMResult dataclass."""

    def test_default_success(self) -> None:
        result = RLMResult(answer="hello", stats=RLMStats())
        assert result.success is True
        assert result.error is None
        assert result.history == []
        assert result.answer == "hello"

    def test_error_case(self) -> None:
        result = RLMResult(
            answer="",
            stats=RLMStats(),
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.answer == ""

    def test_history_is_mutable_list(self) -> None:
        result = RLMResult(answer="x", stats=RLMStats())
        result.history.append({"iteration": 1, "data": "test"})
        assert len(result.history) == 1
        assert result.history[0]["iteration"] == 1


# =====================================================================
# RLM._extract_code_blocks
# =====================================================================


class TestExtractCodeBlocks:
    """Tests for the static _extract_code_blocks method."""

    def test_single_python_block(self) -> None:
        text = '```python\nprint("hello")\n```'
        blocks = RLM._extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]

    def test_multiple_python_blocks(self) -> None:
        text = "Here is code:\n```python\nx = 1\n```\nAnd more:\n```python\ny = 2\n```"
        blocks = RLM._extract_code_blocks(text)
        assert len(blocks) == 2
        assert "x = 1" in blocks[0]
        assert "y = 2" in blocks[1]

    def test_no_code_blocks(self) -> None:
        text = "Just some text with no code."
        blocks = RLM._extract_code_blocks(text)
        assert blocks == []

    def test_non_python_blocks_ignored(self) -> None:
        text = "```bash\necho hello\n```\n```json\n{}\n```"
        blocks = RLM._extract_code_blocks(text)
        assert blocks == []

    def test_untagged_blocks_ignored(self) -> None:
        text = "```\nprint('untagged')\n```"
        blocks = RLM._extract_code_blocks(text)
        assert blocks == []

    def test_code_with_special_characters(self) -> None:
        text = '```python\nresult = re.findall(r"\\d+", text)\nprint(f"Found: {result}")\n```'
        blocks = RLM._extract_code_blocks(text)
        assert len(blocks) == 1
        assert "re.findall" in blocks[0]

    def test_nested_backticks_in_output(self) -> None:
        text = '```python\ncode = "```not a block```"\nprint(code)\n```'
        blocks = RLM._extract_code_blocks(text)
        assert len(blocks) == 1


# =====================================================================
# RLM._create_llm_query_fn
# =====================================================================


class TestCreateLlmQueryFn:
    """Tests for _create_llm_query_fn."""

    def test_returns_callable(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend)
        fn = rlm._create_llm_query_fn("some context")
        assert callable(fn)

    def test_callable_returns_llm_response(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend)
        fn = rlm._create_llm_query_fn("context")
        fn.bind_stats(RLMStats())  # type: ignore[attr-defined]
        result = fn("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_depth_returns_error(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend, max_depth=2)
        fn = rlm._create_llm_query_fn("context", current_depth=2)
        result = fn("any prompt")
        assert "Maximum recursion depth reached" in result

    def test_stats_binding_increments_recursive_calls(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend)
        fn = rlm._create_llm_query_fn("context")
        stats = RLMStats()
        fn.bind_stats(stats)  # type: ignore[attr-defined]
        fn("test prompt")
        assert stats.recursive_calls == 1
        assert stats.llm_calls == 1


# =====================================================================
# RLM._log
# =====================================================================


class TestLog:
    """Tests for _log method."""

    def test_prints_when_verbose(
        self, echo_backend: CallbackBackend, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rlm = RLM(backend=echo_backend, verbose=True)
        rlm._log("test message")
        captured = capsys.readouterr()
        assert "[RLM] test message" in captured.out

    def test_silent_when_not_verbose(
        self, echo_backend: CallbackBackend, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rlm = RLM(backend=echo_backend, verbose=False)
        rlm._log("test message")
        captured = capsys.readouterr()
        assert captured.out == ""


# =====================================================================
# RLM._log_context_size
# =====================================================================


class TestLogContextSize:
    """Tests for _log_context_size."""

    def test_string_context_logs_char_count(
        self, echo_backend: CallbackBackend, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rlm = RLM(backend=echo_backend, verbose=True)
        rlm._log_context_size("hello world")
        captured = capsys.readouterr()
        assert "11" in captured.out
        assert "characters" in captured.out

    def test_path_context_logs_byte_size(
        self,
        echo_backend: CallbackBackend,
        tmp_text_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rlm = RLM(backend=echo_backend, verbose=True)
        rlm._log_context_size(tmp_text_file)
        captured = capsys.readouterr()
        assert "bytes" in captured.out
        assert "memory-mapped" in captured.out

    def test_composite_context_logs_file_count(
        self,
        echo_backend: CallbackBackend,
        composite_context: CompositeContext,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rlm = RLM(backend=echo_backend, verbose=True)
        rlm._log_context_size(composite_context)
        captured = capsys.readouterr()
        assert "files" in captured.out
        assert "bytes" in captured.out
        assert "composite" in captured.out

    def test_list_path_context_logs_file_count(
        self,
        echo_backend: CallbackBackend,
        tmp_multifile_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        paths = sorted(tmp_multifile_dir.glob("*.txt"))
        rlm = RLM(backend=echo_backend, verbose=True)
        rlm._log_context_size(paths)
        captured = capsys.readouterr()
        assert "files" in captured.out
        assert "bytes" in captured.out


# =====================================================================
# RLM._execute_code_blocks
# =====================================================================


class TestExecuteCodeBlocks:
    """Tests for _execute_code_blocks."""

    def _make_rlm(self, backend: CallbackBackend | None = None) -> RLM:
        if backend is None:
            backend = make_deterministic_callback(["unused"])
        return RLM(backend=backend)

    def test_successful_execution(self) -> None:
        from rlm.repl import REPLEnv

        rlm = self._make_rlm()
        repl = REPLEnv(context="test context", llm_query_fn=noop_llm_query)
        outputs, final = rlm._execute_code_blocks(['print("hello")'], repl)
        assert final is None
        assert any("hello" in o for o in outputs)

    def test_error_returns_error_message(self) -> None:
        from rlm.repl import REPLEnv

        rlm = self._make_rlm()
        repl = REPLEnv(context="test context", llm_query_fn=noop_llm_query)
        outputs, final = rlm._execute_code_blocks(["1/0"], repl)
        assert final is None
        assert any("Error" in o for o in outputs)

    def test_final_in_code_block(self) -> None:
        from rlm.repl import REPLEnv

        rlm = self._make_rlm()
        repl = REPLEnv(context="test context", llm_query_fn=noop_llm_query)
        outputs, final = rlm._execute_code_blocks(['FINAL("done")'], repl)
        assert final == "done"

    def test_multiple_blocks_final_in_last(self) -> None:
        from rlm.repl import REPLEnv

        rlm = self._make_rlm()
        repl = REPLEnv(context="test context", llm_query_fn=noop_llm_query)
        code_blocks = ['print("step1")', 'FINAL("the answer")']
        outputs, final = rlm._execute_code_blocks(code_blocks, repl)
        assert final == "the answer"
        assert any("step1" in o for o in outputs)


# =====================================================================
# RLM.completion
# =====================================================================


class TestCompletion:
    """Tests for the full completion loop."""

    def test_successful_completion(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "What is the answer?")
        assert result.success is True
        assert "42" in result.answer

    def test_max_iterations_returns_error(self) -> None:
        # Backend that never calls FINAL — just keeps printing
        backend = make_deterministic_callback(['```python\nprint("looping")\n```'])
        rlm = RLM(backend=backend, max_iterations=3)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is False
        assert "Maximum iterations" in (result.error or "")

    def test_llm_error_returns_error_result(self) -> None:
        def _raise(messages: list[dict[str, str]], model: str) -> str:
            raise RuntimeError("backend exploded")

        backend = CallbackBackend(_raise)
        rlm = RLM(backend=backend, max_iterations=5)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is False
        assert "backend exploded" in (result.error or "")

    def test_no_code_blocks_prompts_for_code(self) -> None:
        # First response has no code, second produces FINAL
        responses = [
            "I will analyze this for you.",
            '```python\nFINAL("got it")\n```',
        ]
        backend = make_deterministic_callback(responses)
        rlm = RLM(backend=backend, max_iterations=5)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert "got it" in result.answer

    def test_stats_populated(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.stats.iterations >= 1
        assert result.stats.llm_calls >= 1


# =====================================================================
# RLM.cost_summary
# =====================================================================


class TestCostSummary:
    """Tests for cost_summary."""

    def test_before_completion_returns_zeros(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend)
        summary = rlm.cost_summary()
        assert summary["iterations"] == 0
        assert summary["llm_calls"] == 0
        assert summary["recursive_calls"] == 0

    def test_after_completion_returns_populated(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        rlm.completion(SAMPLE_TEXT, "query")
        summary = rlm.cost_summary()
        assert summary["iterations"] > 0
        assert summary["llm_calls"] > 0


# =====================================================================
# RLM.acompletion
# =====================================================================


class TestACompletion:
    """Tests for async acompletion."""

    def test_delegates_to_sync(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = asyncio.run(rlm.acompletion(SAMPLE_TEXT, "query"))
        assert result.success is True
        assert "42" in result.answer
