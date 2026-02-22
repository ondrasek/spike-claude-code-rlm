"""End-to-end integration tests for RLM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rlm.backends import AnthropicBackend, OpenAICompatibleBackend
from rlm.cli import (
    _create_backend,
    _create_backend_from_resolved,
    _first_int,
    _first_optional,
    _roles_differ,
    main,
)
from rlm.config import ResolvedRoleConfig
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
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = rlm.completion(SAMPLE_TEXT, "Summarize this document")
        assert result.success is True
        assert len(result.answer) > 0

    def test_file_context(self, tmp_text_file: Path) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = rlm.completion(tmp_text_file, "Summarize this document")
        assert result.success is True
        assert len(result.answer) > 0

    def test_composite_context(self, tmp_multifile_dir: Path) -> None:
        ctx = CompositeContext.from_directory(tmp_multifile_dir)
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
        result = rlm.completion(ctx, "Summarize these documents")
        assert result.success is True
        assert len(result.answer) > 0
        ctx.close()

    def test_verbose_mode_produces_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        backend = make_final_in_two_iterations_callback()
        rlm = RLM(backend=backend, model="test-model", max_iterations=10, verbose=True)
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
            [
                "rlm",
                "--backend",
                "anthropic",
                "--model",
                "test-model",
                "--context-file",
                "/no/such/file.txt",
            ],
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
            patch("sys.argv", ["rlm", "--backend", "openai", "--model", "gpt-4o"]),
            patch.dict("os.environ", {}, clear=False),
        ):
            # Ensure the key is absent even if set in the real env.
            os.environ.pop("OPENAI_API_KEY", None)
            exit_code = main()
        assert exit_code == 1

    def test_openrouter_missing_key_exits_one(self) -> None:
        with (
            patch(
                "sys.argv",
                ["rlm", "--backend", "openrouter", "--model", "anthropic/claude-sonnet-4"],
            ),
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            exit_code = main()
        assert exit_code == 1

    def test_huggingface_missing_key_exits_one(self) -> None:
        with (
            patch(
                "sys.argv",
                ["rlm", "--backend", "huggingface", "--model", "Qwen/Qwen2.5-Coder-32B"],
            ),
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
        rlm = RLM(backend=backend, model="test-model", max_iterations=10)
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
        "model": "test-model",
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
            ("huggingface", "HF_TOKEN", "https://router.huggingface.co/v1"),
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

    def test_model_required(self) -> None:
        """CLI exits with error when --model is not provided."""
        with patch("sys.argv", ["rlm", "--backend", "anthropic"]):
            exit_code = main()
            assert exit_code == 1


# =====================================================================
# _create_backend_from_resolved
# =====================================================================


class TestCreateBackendFromResolved:
    """Tests for _create_backend_from_resolved."""

    def test_anthropic_backend(self) -> None:
        resolved = ResolvedRoleConfig(backend="anthropic", api_key="test-key")
        backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, AnthropicBackend)

    def test_anthropic_falls_back_to_env(self) -> None:
        resolved = ResolvedRoleConfig(backend="anthropic")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, AnthropicBackend)

    def test_anthropic_missing_key_raises(self) -> None:
        resolved = ResolvedRoleConfig(backend="anthropic")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _create_backend_from_resolved(resolved, "root")

    def test_ollama_backend(self) -> None:
        resolved = ResolvedRoleConfig(backend="ollama", base_url="http://localhost:11434/v1")
        backend = _create_backend_from_resolved(resolved, "sub_rlm")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "http://localhost:11434/v1"

    def test_openai_preset(self) -> None:
        resolved = ResolvedRoleConfig(backend="openai", api_key="test-key")
        backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "https://api.openai.com/v1"

    def test_openai_preset_falls_back_to_env(self) -> None:
        resolved = ResolvedRoleConfig(backend="openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, OpenAICompatibleBackend)

    def test_openai_preset_missing_key_raises(self) -> None:
        resolved = ResolvedRoleConfig(backend="openai")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _create_backend_from_resolved(resolved, "root")

    def test_default_backend_is_anthropic(self) -> None:
        resolved = ResolvedRoleConfig(api_key="test-key")
        backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, AnthropicBackend)

    def test_unknown_backend_raises(self) -> None:
        resolved = ResolvedRoleConfig(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown backend"):
            _create_backend_from_resolved(resolved, "root")

    def test_openrouter_preset(self) -> None:
        resolved = ResolvedRoleConfig(backend="openrouter", api_key="test-key")
        backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "https://openrouter.ai/api/v1"

    def test_custom_base_url_for_preset(self) -> None:
        resolved = ResolvedRoleConfig(
            backend="openai", api_key="k", base_url="https://custom.example.com/v1"
        )
        backend = _create_backend_from_resolved(resolved, "root")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.base_url == "https://custom.example.com/v1"


# =====================================================================
# _roles_differ
# =====================================================================


class TestRolesDiffer:
    """Tests for _roles_differ."""

    def test_identical_roles(self) -> None:
        a = ResolvedRoleConfig(backend="anthropic", base_url=None, api_key="k")
        b = ResolvedRoleConfig(backend="anthropic", base_url=None, api_key="k")
        assert not _roles_differ(a, b)

    def test_different_backend(self) -> None:
        a = ResolvedRoleConfig(backend="anthropic")
        b = ResolvedRoleConfig(backend="ollama")
        assert _roles_differ(a, b)

    def test_different_base_url(self) -> None:
        a = ResolvedRoleConfig(backend="ollama", base_url="http://a/v1")
        b = ResolvedRoleConfig(backend="ollama", base_url="http://b/v1")
        assert _roles_differ(a, b)

    def test_different_api_key(self) -> None:
        a = ResolvedRoleConfig(backend="openai", api_key="key1")
        b = ResolvedRoleConfig(backend="openai", api_key="key2")
        assert _roles_differ(a, b)


# =====================================================================
# _first_int, _first_optional helpers
# =====================================================================


class TestFirstHelpers:
    """Tests for _first_int and _first_optional."""

    def test_first_int_returns_first_non_none(self) -> None:
        assert _first_int(None, 5, 10) == 5
        assert _first_int(3, 5, 10) == 3
        assert _first_int(None, None, 10) == 10

    def test_first_optional_returns_first_non_none(self) -> None:
        assert _first_optional(None, 3.14) == 3.14
        assert _first_optional(2.0, 3.14) == 2.0
        assert _first_optional(None, None) is None


# =====================================================================
# Config-aware CLI (--config)
# =====================================================================


def _write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    """Write a YAML config file and return its path."""
    p = tmp_path / "rlm.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


class TestConfigCLI:
    """Tests for the --config CLI path."""

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        config_path = tmp_path / "nonexistent.yaml"
        with patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]):
            exit_code = main()
        assert exit_code == 1

    def test_config_invalid_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("- a\n- b\n", encoding="utf-8")
        with patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]):
            exit_code = main()
        assert exit_code == 1

    def test_config_no_model_exits_one(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, {"defaults": {"backend": "anthropic"}})
        with patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]):
            exit_code = main()
        assert exit_code == 1

    def test_config_missing_api_key_exits_one(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {"root": {"model": "claude-test"}},
            },
        )
        with (
            patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            exit_code = main()
        assert exit_code == 1

    def test_config_settings_merge(self, tmp_path: Path) -> None:
        """Config settings should be picked up when no CLI override."""
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {"root": {"model": "claude-test"}},
                "settings": {"max_iterations": 3, "verbose": True},
            },
        )
        with (
            patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        # Verify the RLM instance received merged settings
        rlm_instance = mock_run.call_args[0][0]
        assert rlm_instance.max_iterations == 3
        assert rlm_instance.verbose is True

    def test_cli_model_overrides_config(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {"root": {"model": "config-model"}},
            },
        )
        with (
            patch(
                "sys.argv",
                ["rlm", "--config", str(config_path), "--model", "cli-model", "--query", "test"],
            ),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        assert rlm_instance.model == "cli-model"

    def test_per_role_backends(self, tmp_path: Path) -> None:
        """Different roles with different backends get distinct backend objects."""
        config_path = _write_config(
            tmp_path,
            {
                "roles": {
                    "root": {"backend": "anthropic", "model": "claude-test"},
                    "sub_rlm": {
                        "backend": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434/v1",
                    },
                },
            },
        )
        with (
            patch(
                "sys.argv",
                ["rlm", "--config", str(config_path), "--query", "test"],
            ),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        assert isinstance(rlm_instance.backend, AnthropicBackend)
        assert isinstance(rlm_instance.sub_rlm_backend, OpenAICompatibleBackend)

    def test_shared_backend_when_configs_match(self, tmp_path: Path) -> None:
        """Same role configs should share the root backend object."""
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {
                    "root": {"model": "claude-test"},
                    "sub_rlm": {"model": "claude-test"},
                },
            },
        )
        with (
            patch(
                "sys.argv",
                ["rlm", "--config", str(config_path), "--query", "test"],
            ),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        # Same backend object should be shared
        assert rlm_instance.sub_rlm_backend is rlm_instance.backend

    def test_custom_system_prompt(self, tmp_path: Path) -> None:
        """Custom system_prompt from config should be passed to RLM."""
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {
                    "root": {"model": "claude-test", "system_prompt": "Custom root prompt"},
                },
            },
        )
        with (
            patch(
                "sys.argv",
                ["rlm", "--config", str(config_path), "--query", "test"],
            ),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        assert rlm_instance.root_system_prompt == "Custom root prompt"

    def test_cli_backend_overrides_config(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            {
                "roles": {
                    "root": {"backend": "openai", "model": "gpt-4o"},
                },
            },
        )
        with (
            patch(
                "sys.argv",
                [
                    "rlm",
                    "--config",
                    str(config_path),
                    "--backend",
                    "anthropic",
                    "--model",
                    "claude-test",
                    "--query",
                    "test",
                ],
            ),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        assert isinstance(rlm_instance.backend, AnthropicBackend)

    def test_config_verify_setting(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            {
                "defaults": {"backend": "anthropic"},
                "roles": {"root": {"model": "claude-test"}},
                "settings": {"verify": True},
            },
        )
        with (
            patch("sys.argv", ["rlm", "--config", str(config_path), "--query", "test"]),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("rlm.cli._run_completion", return_value=0) as mock_run,
        ):
            exit_code = main()
        assert exit_code == 0
        rlm_instance = mock_run.call_args[0][0]
        assert rlm_instance.verify is True
