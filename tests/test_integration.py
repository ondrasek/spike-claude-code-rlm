"""End-to-end integration tests for RLM."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rlm.backends import OpenAICompatibleBackend
from rlm.cli import _BACKEND_DEFAULT_MODELS, _create_backend, main
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

    def test_nonexistent_context_file_exit_one(self) -> None:
        with patch(
            "sys.argv",
            ["rlm", "--backend", "anthropic", "--context-file", "/no/such/file.txt"],
        ):
            exit_code = main()
        assert exit_code == 1

    def test_version_flag(self) -> None:
        with patch("sys.argv", ["rlm", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# =====================================================================
# CLI backend presets
# =====================================================================


class TestCLIBackendPresets:
    """Verify that new backend presets validate API key env vars."""

    def test_openai_missing_key_exits_one(self) -> None:
        with (
            patch("sys.argv", ["rlm", "--backend", "openai"]),
            patch.dict("os.environ", {}, clear=False),
        ):
            # Ensure the key is absent even if set in the real env.
            os.environ.pop("OPENAI_API_KEY", None)
            exit_code = main()
        assert exit_code == 1

    def test_openrouter_missing_key_exits_one(self) -> None:
        with (
            patch("sys.argv", ["rlm", "--backend", "openrouter"]),
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            exit_code = main()
        assert exit_code == 1

    def test_huggingface_missing_key_exits_one(self) -> None:
        with (
            patch("sys.argv", ["rlm", "--backend", "huggingface"]),
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("HF_TOKEN", None)
            exit_code = main()
        assert exit_code == 1


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


# =====================================================================
# Backend factory unit tests
# =====================================================================


def _make_args(**overrides: object) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` that mimics CLI-parsed args."""
    defaults: dict[str, object] = {
        "backend": "anthropic",
        "model": None,
        "base_url": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBackendFactory:
    """Unit tests for ``_create_backend()`` — no live API calls."""

    @pytest.mark.parametrize(
        ("backend_name", "env_var", "expected_url"),
        [
            ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
            ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            ("huggingface", "HF_TOKEN", "https://api-inference.huggingface.co/v1"),
        ],
    )
    def test_creates_openai_compatible_backend(
        self, backend_name: str, env_var: str, expected_url: str
    ) -> None:
        args = _make_args(backend=backend_name)
        with patch.dict(os.environ, {env_var: "test-key-123"}):
            backend = _create_backend(args)
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == expected_url
        assert backend.client.api_key == "test-key-123"

    @pytest.mark.parametrize(
        ("backend_name", "env_var", "expected_url"),
        [
            ("openai", "OPENAI_API_KEY", "https://custom.example.com/v1"),
            ("openrouter", "OPENROUTER_API_KEY", "https://custom.example.com/v1"),
            ("huggingface", "HF_TOKEN", "https://custom.example.com/v1"),
        ],
    )
    def test_base_url_override(self, backend_name: str, env_var: str, expected_url: str) -> None:
        args = _make_args(backend=backend_name, base_url=expected_url)
        with patch.dict(os.environ, {env_var: "key"}):
            backend = _create_backend(args)
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == expected_url

    @pytest.mark.parametrize("backend_name", list(_BACKEND_DEFAULT_MODELS))
    def test_model_defaults_applied(self, backend_name: str) -> None:
        """``_create_backend`` fills in the per-backend default model."""
        args = _make_args(backend=backend_name, model=None)
        # We only care that _resolve_model runs; skip backends that need
        # API keys or external services by catching expected errors.
        with contextlib.suppress(ValueError, FileNotFoundError):
            _create_backend(args)
        assert args.model == _BACKEND_DEFAULT_MODELS[backend_name]
