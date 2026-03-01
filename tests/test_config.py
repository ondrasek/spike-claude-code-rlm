"""Unit tests for rlm/config.py (YAML config loading, validation, resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rlm.config import (
    ConfigError,
    DefaultsConfig,
    ResolvedRoleConfig,
    RLMConfig,
    RoleConfig,
    SettingsConfig,
    load_config,
    resolve_role,
)

# =====================================================================
# Helpers
# =====================================================================


def _write_yaml(tmp_path: Path, data: object, name: str = "rlm.yaml") -> Path:
    """Write a YAML file and return its path."""
    path = tmp_path / name
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _write_text(tmp_path: Path, text: str, name: str = "rlm.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# =====================================================================
# load_config — basic parsing
# =====================================================================


class TestLoadConfig:
    """Tests for load_config."""

    def test_minimal_config(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, {"defaults": {"backend": "anthropic"}})
        config = load_config(path)
        assert config.defaults.backend == "anthropic"
        assert config.roles == {}
        assert config.settings.max_iterations is None

    def test_full_config(self, tmp_path: Path) -> None:
        data = {
            "defaults": {"backend": "anthropic", "model": "claude-sonnet"},
            "roles": {
                "root": {"model": "claude-opus"},
                "sub_rlm": {"backend": "ollama", "model": "llama3.2"},
                "verifier": {"model": "claude-haiku"},
            },
            "settings": {
                "max_iterations": 5,
                "max_depth": 2,
                "verbose": True,
                "verify": True,
            },
        }
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        assert config.defaults.model == "claude-sonnet"
        assert config.roles["root"].model == "claude-opus"
        assert config.roles["sub_rlm"].backend == "ollama"
        assert config.settings.max_iterations == 5
        assert config.settings.verify is True

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "- a\n- b\n")
        with pytest.raises(ConfigError, match="YAML mapping"):
            load_config(path)

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "")
        config = load_config(path)
        assert config.defaults.backend is None
        assert config.roles == {}

    def test_config_dir_set_to_parent(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        path = _write_yaml(subdir, {"defaults": {}})
        config = load_config(path)
        assert config.config_dir == subdir


# =====================================================================
# Validation
# =====================================================================


class TestValidation:
    """Tests for _validate_config (called by load_config)."""

    def test_unknown_role_name_raises(self, tmp_path: Path) -> None:
        data = {"roles": {"bogus": {"model": "x"}}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Unknown role 'bogus'"):
            load_config(path)

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        data = {"defaults": {"backend": "deepseek"}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Unknown backend 'deepseek'"):
            load_config(path)

    def test_unknown_backend_in_role_raises(self, tmp_path: Path) -> None:
        data = {"roles": {"root": {"backend": "nonexistent"}}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Unknown backend 'nonexistent'"):
            load_config(path)

    def test_both_prompt_and_file_raises(self, tmp_path: Path) -> None:
        data = {
            "roles": {
                "root": {
                    "system_prompt": "inline prompt",
                    "system_prompt_file": "prompt.txt",
                }
            }
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="both system_prompt and system_prompt_file"):
            load_config(path)

    def test_missing_prompt_file_raises(self, tmp_path: Path) -> None:
        data = {"roles": {"root": {"system_prompt_file": "missing.txt"}}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="system_prompt_file not found"):
            load_config(path)

    def test_valid_config_passes(self, tmp_path: Path) -> None:
        data = {
            "defaults": {"backend": "anthropic", "model": "claude-sonnet"},
            "roles": {
                "root": {"model": "claude-opus"},
                "sub_rlm": {"backend": "ollama", "model": "llama3.2"},
            },
            "settings": {"verbose": True},
        }
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        assert config.defaults.backend == "anthropic"


# =====================================================================
# resolve_role
# =====================================================================


class TestResolveRole:
    """Tests for resolve_role."""

    def test_role_overrides_defaults(self) -> None:
        config = RLMConfig(
            defaults=DefaultsConfig(backend="anthropic", model="default-model"),
            roles={"root": RoleConfig(model="custom-model")},
        )
        resolved = resolve_role("root", config)
        assert resolved.model == "custom-model"
        assert resolved.backend == "anthropic"  # from defaults

    def test_defaults_used_when_role_absent(self) -> None:
        config = RLMConfig(
            defaults=DefaultsConfig(backend="ollama", model="llama3.2"),
        )
        resolved = resolve_role("sub_rlm", config)
        assert resolved.backend == "ollama"
        assert resolved.model == "llama3.2"

    def test_prompt_file_loaded(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("Custom prompt text", encoding="utf-8")
        config = RLMConfig(
            roles={"root": RoleConfig(system_prompt_file="custom.txt")},
            config_dir=tmp_path,
        )
        resolved = resolve_role("root", config)
        assert resolved.system_prompt == "Custom prompt text"

    def test_inline_prompt_used(self) -> None:
        config = RLMConfig(
            roles={"verifier": RoleConfig(system_prompt="Be strict.")},
        )
        resolved = resolve_role("verifier", config)
        assert resolved.system_prompt == "Be strict."

    def test_paths_relative_to_config_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "prompts"
        subdir.mkdir()
        (subdir / "root.txt").write_text("Root prompt", encoding="utf-8")
        config = RLMConfig(
            roles={"root": RoleConfig(system_prompt_file="prompts/root.txt")},
            config_dir=tmp_path,
        )
        resolved = resolve_role("root", config)
        assert resolved.system_prompt == "Root prompt"

    def test_api_key_env_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TEST_KEY", "secret123")
        config = RLMConfig(
            defaults=DefaultsConfig(api_key_env="MY_TEST_KEY"),
        )
        resolved = resolve_role("root", config)
        assert resolved.api_key == "secret123"

    def test_missing_env_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        config = RLMConfig(
            defaults=DefaultsConfig(api_key_env="NONEXISTENT_KEY"),
        )
        resolved = resolve_role("root", config)
        assert resolved.api_key is None

    def test_no_prompt_returns_none(self) -> None:
        config = RLMConfig(
            roles={"root": RoleConfig(model="x")},
        )
        resolved = resolve_role("root", config)
        assert resolved.system_prompt is None


# =====================================================================
# Merge priority (simulating CLI > config interaction)
# =====================================================================


class TestMergePriority:
    """Tests simulating the CLI > config merge logic from cli.py."""

    def test_cli_backend_overrides_config(self) -> None:
        config = RLMConfig(
            defaults=DefaultsConfig(backend="anthropic"),
            roles={"root": RoleConfig(backend="ollama")},
        )
        resolved = resolve_role("root", config)
        assert resolved.backend == "ollama"  # from role

        # Simulate CLI override
        resolved.backend = "openai"
        assert resolved.backend == "openai"

    def test_cli_model_overrides_config(self) -> None:
        config = RLMConfig(
            roles={"root": RoleConfig(model="config-model")},
        )
        resolved = resolve_role("root", config)
        assert resolved.model == "config-model"

        # Simulate CLI override
        resolved.model = "cli-model"
        assert resolved.model == "cli-model"

    def test_config_model_used_when_no_cli(self) -> None:
        config = RLMConfig(
            defaults=DefaultsConfig(model="default-model"),
            roles={"root": RoleConfig(model="root-model")},
        )
        resolved = resolve_role("root", config)
        assert resolved.model == "root-model"


# =====================================================================
# Dataclass defaults
# =====================================================================


class TestDataclassDefaults:
    """Tests for dataclass default values."""

    def test_defaults_config_all_none(self) -> None:
        d = DefaultsConfig()
        assert d.backend is None
        assert d.model is None
        assert d.base_url is None
        assert d.api_key_env is None

    def test_role_config_all_none(self) -> None:
        r = RoleConfig()
        assert r.backend is None
        assert r.system_prompt is None
        assert r.system_prompt_file is None

    def test_settings_config_all_none(self) -> None:
        s = SettingsConfig()
        assert s.max_iterations is None
        assert s.verbose is None
        assert s.verify is None

    def test_settings_config_structured_output_default_none(self) -> None:
        s = SettingsConfig()
        assert s.structured_output is None

    def test_resolved_role_config_all_none(self) -> None:
        r = ResolvedRoleConfig()
        assert r.backend is None
        assert r.model is None
        assert r.api_key is None
        assert r.system_prompt is None


# =====================================================================
# Structured output config validation
# =====================================================================


class TestStructuredOutputConfig:
    """Tests for structured_output config validation."""

    def test_valid_modes_accepted(self, tmp_path: Path) -> None:
        for mode in ("probe", "on", "off"):
            data = {"settings": {"structured_output": mode}}
            path = _write_yaml(tmp_path, data, name=f"rlm_{mode}.yaml")
            config = load_config(path)
            assert config.settings.structured_output == mode

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        data = {"settings": {"structured_output": "always"}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Unknown structured_output 'always'"):
            load_config(path)

    def test_default_is_none(self) -> None:
        s = SettingsConfig()
        assert s.structured_output is None
