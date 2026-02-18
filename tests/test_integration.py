"""End-to-end integration tests for RLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rlm.cli import main
from rlm.context import CompositeContext
from rlm.rlm import RLM

from .conftest import (
    SAMPLE_TEXT,
    make_final_in_two_iterations_callback,
)

# =====================================================================
# Full RLM loop
# =====================================================================


class TestFullRLMLoop:
    """End-to-end tests exercising the full RLM orchestrator loop."""

    def test_string_context(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "Summarize this document")
        assert result.success is True
        assert len(result.answer) > 0

    def test_file_context(self, tmp_text_file: Path) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = rlm.completion(tmp_text_file, "Summarize this document")
        assert result.success is True
        assert len(result.answer) > 0

    def test_composite_context(self, tmp_multifile_dir: Path) -> None:
        ctx = CompositeContext.from_directory(tmp_multifile_dir)
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        result = rlm.completion(ctx, "Summarize these documents")
        assert result.success is True
        assert len(result.answer) > 0
        ctx.close()

    def test_verbose_mode_produces_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10, verbose=True)
        rlm.completion(SAMPLE_TEXT, "Summarize")
        captured = capsys.readouterr()
        assert "[RLM]" in captured.out


# =====================================================================
# CLI integration
# =====================================================================


class TestCLIIntegration:
    """Tests for the CLI main() entry point."""

    def test_callback_backend_exit_zero(self) -> None:
        with patch("sys.argv", ["rlm", "--backend", "callback"]):
            exit_code = main()
        assert exit_code == 0

    def test_callback_verbose_prints_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["rlm", "--backend", "callback", "--verbose"]):
            exit_code = main()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "FINAL ANSWER" in captured.out

    def test_nonexistent_context_file_exit_one(self) -> None:
        with patch(
            "sys.argv",
            ["rlm", "--backend", "callback", "--context-file", "/no/such/file.txt"],
        ):
            exit_code = main()
        assert exit_code == 1

    def test_version_flag(self) -> None:
        with patch("sys.argv", ["rlm", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# =====================================================================
# Stats verification
# =====================================================================


class TestStatsVerification:
    """Verify that stats are correctly populated after completion."""

    def test_cost_summary_nonzero_after_completion(self) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, max_iterations=10)
        rlm.completion(SAMPLE_TEXT, "query")
        summary = rlm.cost_summary()
        assert summary["iterations"] > 0
        assert summary["llm_calls"] > 0
