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
        rlm = RLM(backend=echo_backend, model="test-model")
        fn = rlm._create_llm_query_fn("some context")
        assert callable(fn)

    def test_callable_returns_llm_response(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend, model="test-model")
        fn = rlm._create_llm_query_fn("context")
        fn.bind_stats(RLMStats())  # type: ignore[attr-defined]
        result = fn("snippet", "task")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_depth_returns_error(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend, model="test-model", max_depth=2)
        fn = rlm._create_llm_query_fn("context", current_depth=2)
        result = fn("snippet", "any task")
        assert "Maximum recursion depth reached" in result

    def test_stats_binding_increments_recursive_calls(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend, model="test-model")
        fn = rlm._create_llm_query_fn("context")
        stats = RLMStats()
        fn.bind_stats(stats)  # type: ignore[attr-defined]
        fn("snippet", "test task")
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
        rlm = RLM(backend=echo_backend, model="test-model", verbose=True)
        rlm._log("test message")
        captured = capsys.readouterr()
        assert "[RLM] test message" in captured.out

    def test_silent_when_not_verbose(
        self, echo_backend: CallbackBackend, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rlm = RLM(backend=echo_backend, model="test-model", verbose=False)
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
        rlm = RLM(backend=echo_backend, model="test-model", verbose=True)
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
        rlm = RLM(backend=echo_backend, model="test-model", verbose=True)
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
        rlm = RLM(backend=echo_backend, model="test-model", verbose=True)
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
        rlm = RLM(backend=echo_backend, model="test-model", verbose=True)
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
        return RLM(backend=backend, model="test-model")

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
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "What is the answer?")
        assert result.success is True
        assert "42" in result.answer

    def test_max_iterations_returns_error(self) -> None:
        # Backend that never calls FINAL — just keeps printing
        backend = make_deterministic_callback(['```python\nprint("looping")\n```'])
        rlm = RLM(backend=backend, model="test-model", max_iterations=3)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is False
        assert "Maximum iterations" in (result.error or "")

    def test_llm_error_returns_error_result(self) -> None:
        def _raise(messages: list[dict[str, str]], model: str) -> str:
            raise RuntimeError("backend exploded")

        backend = CallbackBackend(_raise)
        rlm = RLM(backend=backend, model="test-model", max_iterations=5)
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
        rlm = RLM(backend=backend, model="test-model", max_iterations=5)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert "got it" in result.answer

    def test_stats_populated(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.stats.iterations >= 1
        assert result.stats.llm_calls >= 1


# =====================================================================
# RLM.cost_summary
# =====================================================================


class TestCostSummary:
    """Tests for cost_summary."""

    def test_before_completion_returns_zeros(self, echo_backend: CallbackBackend) -> None:
        rlm = RLM(backend=echo_backend, model="test-model")
        summary = rlm.cost_summary()
        assert summary["iterations"] == 0
        assert summary["llm_calls"] == 0
        assert summary["recursive_calls"] == 0

    def test_after_completion_returns_populated(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
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
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = asyncio.run(rlm.acompletion(SAMPLE_TEXT, "query"))
        assert result.success is True
        assert "42" in result.answer


# =====================================================================
# RLM.include_context_sample
# =====================================================================


class TestNoContextSample:
    """Tests for the include_context_sample flag."""

    def test_context_sample_skipped_when_disabled(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=10,
            include_context_sample=False,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # Verify the answer still works — the LLM loop completed
        assert len(result.answer) > 0


# =====================================================================
# RLM.timeout and max_token_budget
# =====================================================================


class TestTimeoutGuard:
    """Tests for the timeout guard."""

    def test_timeout_returns_error(self) -> None:
        """A tiny timeout should trip before the loop finishes."""
        import time

        call_count = {"n": 0}

        def _slow(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            time.sleep(0.1)
            return '```python\nprint("still going")\n```'

        backend = CallbackBackend(_slow)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=50,
            timeout=0.05,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is False
        assert "Timeout" in (result.error or "")


class TestTokenBudgetGuard:
    """Tests for the max_token_budget guard."""

    def test_token_budget_returns_error(self) -> None:
        """Token budget should trip when usage exceeds the limit."""
        from rlm.backends import CompletionResult, TokenUsage

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            return '```python\nprint("running")\n```'

        # Create a backend that reports non-zero token usage
        class TokenCountingBackend(CallbackBackend):
            def completion(
                self, messages: list[dict[str, str]], model: str, **kwargs: object
            ) -> CompletionResult:
                return CompletionResult(
                    text=self.callback_fn(messages, model),
                    usage=TokenUsage(input_tokens=500, output_tokens=500),
                )

        backend = TokenCountingBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=50,
            max_token_budget=100,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is False
        assert "Token budget exceeded" in (result.error or "")
