"""YAML-based configuration for per-role LLM settings.

Loads an optional ``--config rlm.yaml`` file and resolves per-role backend,
model, base_url, api_key, and system_prompt values using the merge priority::

    CLI flags  >  roles.{role}  >  defaults  >  hardcoded defaults

This module imports only stdlib + ``yaml`` — no ``rlm.*`` imports.
It is consumed exclusively by ``rlm/cli.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised on configuration validation failures."""


# =====================================================================
# Dataclasses
# =====================================================================

_VALID_ROLE_NAMES = frozenset({"root", "sub_rlm", "verifier", "context_engineer"})
_VALID_BACKENDS = frozenset(
    {"anthropic", "openai", "openrouter", "huggingface", "ollama", "claude"}
)
_VALID_CE_MODES = frozenset({"off", "pre_loop", "per_query", "both"})
_VALID_STRUCTURED_OUTPUT_MODES = frozenset({"probe", "on", "off"})


@dataclass
class DefaultsConfig:
    """Shared fallback values for all roles.

    Only backend/model/base_url/api_key_env are allowed here —
    system_prompt fields are role-level only.
    """

    backend: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass
class RoleConfig:
    """Per-role configuration (root, sub_rlm, verifier, context_engineer)."""

    backend: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    per_query_prompt: str | None = None
    per_query_prompt_file: str | None = None


@dataclass
class SettingsConfig:
    """Non-role RLM settings."""

    max_iterations: int | None = None
    max_depth: int | None = None
    max_tokens: int | None = None
    verbose: bool | None = None
    compact: bool | None = None
    timeout: int | None = None
    max_token_budget: int | None = None
    verify: bool | None = None
    context_engineer_mode: str | None = None
    share_brief_with_root: bool | None = None
    structured_output: str | None = None


@dataclass
class RLMConfig:
    """Top-level parsed config file."""

    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    config_dir: Path = field(default_factory=Path.cwd)


@dataclass
class ResolvedRoleConfig:
    """Fully resolved configuration for a single role (after merge)."""

    backend: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    per_query_prompt: str | None = None


# =====================================================================
# Loading & Validation
# =====================================================================


def load_config(path: Path) -> RLMConfig:
    """Parse a YAML config file and return an ``RLMConfig``.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    RLMConfig
        Parsed configuration.

    Raises
    ------
    ConfigError
        If the file is missing, not valid YAML, or fails validation.
    FileNotFoundError
        If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)

    if raw is None:
        # Empty YAML file — return default config
        return RLMConfig(config_dir=path.parent.resolve())

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    config = _parse_raw(raw, config_dir=path.parent.resolve())
    _validate_config(config)
    return config


def _parse_raw(raw: dict[str, Any], config_dir: Path) -> RLMConfig:
    """Build an ``RLMConfig`` from raw YAML dict."""
    defaults = DefaultsConfig()
    if "defaults" in raw and isinstance(raw["defaults"], dict):
        d = raw["defaults"]
        defaults = DefaultsConfig(
            backend=d.get("backend"),
            model=d.get("model"),
            base_url=d.get("base_url"),
            api_key_env=d.get("api_key_env"),
        )

    roles: dict[str, RoleConfig] = {}
    if "roles" in raw and isinstance(raw["roles"], dict):
        for role_name, role_dict in raw["roles"].items():
            if not isinstance(role_dict, dict):
                raise ConfigError(f"Role '{role_name}' must be a mapping")
            roles[role_name] = RoleConfig(
                backend=role_dict.get("backend"),
                model=role_dict.get("model"),
                base_url=role_dict.get("base_url"),
                api_key_env=role_dict.get("api_key_env"),
                system_prompt=role_dict.get("system_prompt"),
                system_prompt_file=role_dict.get("system_prompt_file"),
                per_query_prompt=role_dict.get("per_query_prompt"),
                per_query_prompt_file=role_dict.get("per_query_prompt_file"),
            )

    settings = SettingsConfig()
    if "settings" in raw and isinstance(raw["settings"], dict):
        s = raw["settings"]
        settings = SettingsConfig(
            max_iterations=s.get("max_iterations"),
            max_depth=s.get("max_depth"),
            max_tokens=s.get("max_tokens"),
            verbose=s.get("verbose"),
            compact=s.get("compact"),
            timeout=s.get("timeout"),
            max_token_budget=s.get("max_token_budget"),
            verify=s.get("verify"),
            context_engineer_mode=s.get("context_engineer_mode"),
            share_brief_with_root=s.get("share_brief_with_root"),
            structured_output=s.get("structured_output"),
        )

    return RLMConfig(defaults=defaults, roles=roles, settings=settings, config_dir=config_dir)


def _validate_prompt_pair(
    role_name: str,
    inline: str | None,
    file_field: str | None,
    field_name: str,
    config_dir: Path,
) -> None:
    """Validate that only one of inline/file prompt is set, and file exists."""
    if inline and file_field:
        raise ConfigError(
            f"Role '{role_name}' specifies both {field_name} and {field_name}_file. Use only one."
        )
    if file_field:
        prompt_path = config_dir / file_field
        if not prompt_path.exists():
            raise ConfigError(f"Role '{role_name}' {field_name}_file not found: {prompt_path}")


def _validate_config(config: RLMConfig) -> None:
    """Validate a parsed config, raising ``ConfigError`` on problems."""
    # Check role names
    for role_name in config.roles:
        if role_name not in _VALID_ROLE_NAMES:
            raise ConfigError(
                f"Unknown role '{role_name}'. Valid roles: {sorted(_VALID_ROLE_NAMES)}"
            )

    # Check backend names
    for label, backend in _all_backends(config):
        if backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"Unknown backend '{backend}' in {label}. Valid backends: {sorted(_VALID_BACKENDS)}"
            )

    # Check context_engineer_mode
    ce_mode = config.settings.context_engineer_mode
    if ce_mode is not None and ce_mode not in _VALID_CE_MODES:
        raise ConfigError(
            f"Unknown context_engineer_mode '{ce_mode}'. Valid modes: {sorted(_VALID_CE_MODES)}"
        )

    # Check structured_output mode
    so_mode = config.settings.structured_output
    if so_mode is not None and so_mode not in _VALID_STRUCTURED_OUTPUT_MODES:
        raise ConfigError(
            f"Unknown structured_output '{so_mode}'. "
            f"Valid modes: {sorted(_VALID_STRUCTURED_OUTPUT_MODES)}"
        )

    # Check prompt constraints per role
    for role_name, role in config.roles.items():
        _validate_prompt_pair(
            role_name,
            role.system_prompt,
            role.system_prompt_file,
            "system_prompt",
            config.config_dir,
        )
        _validate_prompt_pair(
            role_name,
            role.per_query_prompt,
            role.per_query_prompt_file,
            "per_query_prompt",
            config.config_dir,
        )


def _all_backends(config: RLMConfig) -> list[tuple[str, str]]:
    """Collect all explicitly set backend values for validation."""
    result: list[tuple[str, str]] = []
    if config.defaults.backend:
        result.append(("defaults", config.defaults.backend))
    for role_name, role in config.roles.items():
        if role.backend:
            result.append((f"roles.{role_name}", role.backend))
    return result


# =====================================================================
# Resolution
# =====================================================================


def resolve_role(role_name: str, config: RLMConfig) -> ResolvedRoleConfig:
    """Merge role config with defaults and resolve dynamic values.

    Parameters
    ----------
    role_name : str
        One of ``"root"``, ``"sub_rlm"``, ``"verifier"``.
    config : RLMConfig
        The parsed config.

    Returns
    -------
    ResolvedRoleConfig
        Fully resolved configuration for the role.
    """
    role = config.roles.get(role_name, RoleConfig())

    # Merge: role > defaults (for backend/model/base_url/api_key_env only)
    backend = role.backend or config.defaults.backend
    model = role.model or config.defaults.model
    base_url = role.base_url or config.defaults.base_url
    api_key_env = role.api_key_env or config.defaults.api_key_env

    # Resolve API key from environment variable
    api_key: str | None = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)

    # Resolve system prompt (role-level only, no defaults)
    system_prompt: str | None = None
    if role.system_prompt:
        system_prompt = role.system_prompt
    elif role.system_prompt_file:
        prompt_path = config.config_dir / role.system_prompt_file
        system_prompt = prompt_path.read_text(encoding="utf-8")

    # Resolve per-query prompt (context_engineer role only, but harmless on others)
    per_query_prompt: str | None = None
    if role.per_query_prompt:
        per_query_prompt = role.per_query_prompt
    elif role.per_query_prompt_file:
        prompt_path = config.config_dir / role.per_query_prompt_file
        per_query_prompt = prompt_path.read_text(encoding="utf-8")

    return ResolvedRoleConfig(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        per_query_prompt=per_query_prompt,
    )
