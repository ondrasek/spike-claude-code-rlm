"""Unit tests for the context-engineer feature."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rlm.backends import CallbackBackend
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
from rlm.prompts import (
    get_context_engineer_per_query_prompt,
    get_context_engineer_pre_loop_prompt,
)
from rlm.rlm import RLM, RLMStats

from .conftest import SAMPLE_TEXT, make_deterministic_callback

# =====================================================================
# Helpers
# =====================================================================


def _write_yaml(tmp_path: Path, data: object, name: str = "rlm.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _make_ce_backend(responses: list[str] | None = None) -> CallbackBackend:
    """Create a callback backend for context-engineer testing."""
    captured: list[list[dict[str, str]]] = []

    def _cb(messages: list[dict[str, str]], model: str) -> str:
        captured.append(messages)
        if responses:
            return responses[len(captured) - 1 % len(responses)]
        return "Mock CE response"

    backend = CallbackBackend(_cb)
    backend._captured = captured  # type: ignore[attr-defined]
    return backend


# =====================================================================
# Config: context_engineer_mode validation
# =====================================================================


class TestConfigCEMode:
    """Tests for context_engineer_mode config validation."""

    def test_valid_ce_modes_accepted(self, tmp_path: Path) -> None:
        for mode in ("off", "pre_loop", "per_query", "both"):
            data = {"settings": {"context_engineer_mode": mode}}
            path = _write_yaml(tmp_path, data, f"rlm_{mode}.yaml")
            config = load_config(path)
            assert config.settings.context_engineer_mode == mode

    def test_invalid_ce_mode_raises(self, tmp_path: Path) -> None:
        data = {"settings": {"context_engineer_mode": "always"}}
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Unknown context_engineer_mode 'always'"):
            load_config(path)

    def test_context_engineer_role_accepted(self, tmp_path: Path) -> None:
        data = {
            "roles": {
                "context_engineer": {"backend": "ollama", "model": "llama3.2"},
            }
        }
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        assert "context_engineer" in config.roles
        assert config.roles["context_engineer"].model == "llama3.2"

    def test_share_brief_with_root_parsed(self, tmp_path: Path) -> None:
        data = {"settings": {"share_brief_with_root": True}}
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        assert config.settings.share_brief_with_root is True

    def test_settings_defaults_none(self) -> None:
        s = SettingsConfig()
        assert s.context_engineer_mode is None
        assert s.share_brief_with_root is None


# =====================================================================
# Config: per_query_prompt fields
# =====================================================================


class TestConfigPerQueryPrompt:
    """Tests for per_query_prompt and per_query_prompt_file config fields."""

    def test_per_query_prompt_inline(self, tmp_path: Path) -> None:
        data = {
            "roles": {
                "context_engineer": {"per_query_prompt": "Custom per-query prompt"},
            }
        }
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        assert config.roles["context_engineer"].per_query_prompt == "Custom per-query prompt"

    def test_per_query_prompt_file(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "pq.txt"
        prompt_file.write_text("Per-query from file", encoding="utf-8")
        data = {
            "roles": {
                "context_engineer": {"per_query_prompt_file": "pq.txt"},
            }
        }
        path = _write_yaml(tmp_path, data)
        config = load_config(path)
        resolved = resolve_role("context_engineer", config)
        assert resolved.per_query_prompt == "Per-query from file"

    def test_both_per_query_prompt_and_file_raises(self, tmp_path: Path) -> None:
        data = {
            "roles": {
                "context_engineer": {
                    "per_query_prompt": "inline",
                    "per_query_prompt_file": "pq.txt",
                },
            }
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="both per_query_prompt and per_query_prompt_file"):
            load_config(path)

    def test_missing_per_query_prompt_file_raises(self, tmp_path: Path) -> None:
        data = {
            "roles": {
                "context_engineer": {"per_query_prompt_file": "missing.txt"},
            }
        }
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="per_query_prompt_file not found"):
            load_config(path)

    def test_resolved_per_query_prompt_default_none(self) -> None:
        r = ResolvedRoleConfig()
        assert r.per_query_prompt is None

    def test_role_config_per_query_fields_default_none(self) -> None:
        r = RoleConfig()
        assert r.per_query_prompt is None
        assert r.per_query_prompt_file is None


# =====================================================================
# Config: resolve_role for context_engineer
# =====================================================================


class TestResolveRoleCE:
    """Tests for resolve_role with the context_engineer role."""

    def test_ce_inherits_defaults(self) -> None:
        config = RLMConfig(
            defaults=DefaultsConfig(backend="ollama", model="llama3.2"),
        )
        resolved = resolve_role("context_engineer", config)
        assert resolved.backend == "ollama"
        assert resolved.model == "llama3.2"

    def test_ce_inline_system_prompt(self) -> None:
        config = RLMConfig(
            roles={"context_engineer": RoleConfig(system_prompt="Custom CE prompt")},
        )
        resolved = resolve_role("context_engineer", config)
        assert resolved.system_prompt == "Custom CE prompt"

    def test_ce_inline_per_query_prompt(self) -> None:
        config = RLMConfig(
            roles={"context_engineer": RoleConfig(per_query_prompt="Custom PQ prompt")},
        )
        resolved = resolve_role("context_engineer", config)
        assert resolved.per_query_prompt == "Custom PQ prompt"


# =====================================================================
# Prompts: getter functions
# =====================================================================


class TestCEPrompts:
    """Tests for context-engineer prompt getter functions."""

    def test_pre_loop_prompt_not_empty(self) -> None:
        prompt = get_context_engineer_pre_loop_prompt()
        assert len(prompt) > 100
        assert "document" in prompt.lower()
        assert "brief" in prompt.lower()

    def test_per_query_prompt_not_empty(self) -> None:
        prompt = get_context_engineer_per_query_prompt()
        assert len(prompt) > 100
        assert "context" in prompt.lower()
        assert "snippet" in prompt.lower()

    def test_pre_loop_prompt_contains_key_instructions(self) -> None:
        prompt = get_context_engineer_pre_loop_prompt()
        assert "terminology" in prompt.lower()
        assert "structure" in prompt.lower()
        assert "200-500" in prompt

    def test_per_query_prompt_contains_key_instructions(self) -> None:
        prompt = get_context_engineer_per_query_prompt()
        assert "50-150" in prompt
        assert "position" in prompt.lower()


# =====================================================================
# RLMStats: context_engineer_calls
# =====================================================================


class TestRLMStatsCE:
    """Tests for context_engineer_calls in RLMStats."""

    def test_default_zero(self) -> None:
        stats = RLMStats()
        assert stats.context_engineer_calls == 0

    def test_custom_initialization(self) -> None:
        stats = RLMStats(context_engineer_calls=3)
        assert stats.context_engineer_calls == 3

    def test_field_mutation(self) -> None:
        stats = RLMStats()
        stats.context_engineer_calls = 5
        assert stats.context_engineer_calls == 5


# =====================================================================
# RLM: mode=off (default — no CE calls)
# =====================================================================


class TestCEModeOff:
    """When mode=off, no context-engineer calls should be made."""

    def test_no_ce_calls_by_default(self) -> None:
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return '```python\nprint("exploring")\n```'
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(backend=backend, model="test-model", max_iterations=5)
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert result.stats.context_engineer_calls == 0

    def test_mode_off_explicit(self) -> None:
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return '```python\nprint("exploring")\n```'
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="off",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert result.stats.context_engineer_calls == 0

    def test_cost_summary_includes_ce_calls(self) -> None:
        responses = [
            '```python\nFINAL("answer")\n```',
        ]
        backend = make_deterministic_callback(responses)
        rlm = RLM(backend=backend, model="test-model", max_iterations=5)
        rlm.completion(SAMPLE_TEXT, "query")
        summary = rlm.cost_summary()
        assert "context_engineer_calls" in summary
        assert summary["context_engineer_calls"] == 0

    def test_cost_summary_before_completion(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        rlm = RLM(backend=backend, model="test-model")
        summary = rlm.cost_summary()
        assert summary["context_engineer_calls"] == 0


# =====================================================================
# RLM: mode=pre_loop
# =====================================================================


class TestCEModePreLoop:
    """Tests for context_engineer_mode=pre_loop."""

    def test_pre_loop_makes_one_ce_call(self) -> None:
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # CE pre-loop call
                return "This is a research paper about AI."
            if call_count["n"] == 2:
                return '```python\nprint("exploring")\n```'
            return '```python\nFINAL("result")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert result.stats.context_engineer_calls == 1

    def test_pre_loop_brief_included_in_sub_rlm_calls(self) -> None:
        """The document brief should appear in sub-RLM user messages."""
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            captured_messages.append(messages)
            if call_count["n"] == 1:
                # CE pre-loop call
                return "BRIEF: This is a legal contract about software licensing."
            if call_count["n"] == 2:
                # Root LM — calls llm_query
                return (
                    "```python\n"
                    'result = llm_query(CONTEXT[:500], "Summarize this section")\n'
                    "print(result)\n"
                    "```"
                )
            if call_count["n"] == 3:
                # Sub-RLM call — should include brief
                return "Section summary"
            if call_count["n"] == 4:
                return '```python\nFINAL("done")\n```'
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # The sub-RLM call (call #3) should include the document brief
        assert len(captured_messages) >= 3
        sub_rlm_msg = captured_messages[2]
        user_content = sub_rlm_msg[-1]["content"]
        assert "Document Brief" in user_content
        assert "legal contract" in user_content

    def test_share_brief_with_root_true(self) -> None:
        """With share_brief_with_root=True, the brief should appear in root LM prompt."""
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            # Snapshot the messages to avoid mutation after the callback
            captured_messages.append([dict(m) for m in messages])
            if call_count["n"] == 1:
                return "BRIEF: Technical specification document."
            return '```python\nFINAL("answer")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
            share_brief_with_root=True,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # The root LM call (call #2) should have the brief in user prompt
        root_msg = captured_messages[1]
        user_content = root_msg[1]["content"]  # index 1 = user message
        assert "Document Brief" in user_content
        assert "Technical specification" in user_content

    def test_share_brief_with_root_false(self) -> None:
        """With share_brief_with_root=False, the brief should NOT appear in root LM prompt."""
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            # Snapshot the messages to avoid mutation after the callback
            captured_messages.append([dict(m) for m in messages])
            if call_count["n"] == 1:
                return "BRIEF: Technical specification document."
            return '```python\nFINAL("answer")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
            share_brief_with_root=False,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # The root LM call (call #2) should NOT have the brief in user prompt
        root_msg = captured_messages[1]
        user_content = root_msg[1]["content"]  # index 1 = user message
        assert "Document Brief" not in user_content


# =====================================================================
# RLM: mode=per_query
# =====================================================================


class TestCEModePerQuery:
    """Tests for context_engineer_mode=per_query."""

    def test_per_query_makes_ce_call_on_llm_query(self) -> None:
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Root LM — calls llm_query
                return (
                    '```python\nresult = llm_query(CONTEXT[:200], "Summarize")\nprint(result)\n```'
                )
            if call_count["n"] == 2:
                # CE per-query call
                return "Context note: This is from the introduction."
            if call_count["n"] == 3:
                # Sub-RLM call
                return "Summary of introduction"
            if call_count["n"] == 4:
                return '```python\nFINAL("done")\n```'
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="per_query",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert result.stats.context_engineer_calls == 1

    def test_per_query_no_ce_call_without_llm_query(self) -> None:
        """If no llm_query() is called, no CE per-query calls should happen."""
        responses = [
            '```python\nprint("exploring")\n```',
            '```python\nFINAL("no sub-rlm needed")\n```',
        ]
        backend = make_deterministic_callback(responses)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="per_query",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert result.stats.context_engineer_calls == 0

    def test_per_query_context_note_in_sub_rlm_message(self) -> None:
        """The context note should appear in the sub-RLM user message."""
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            captured_messages.append(messages)
            if call_count["n"] == 1:
                return (
                    "```python\n"
                    'result = llm_query(CONTEXT[:100], "Extract key points")\n'
                    "print(result)\n"
                    "```"
                )
            if call_count["n"] == 2:
                # CE per-query
                return "NOTE: This comes from Chapter 1 introduction."
            if call_count["n"] == 3:
                # Sub-RLM
                return "Key points extracted"
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="per_query",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # Sub-RLM message (call #3) should include context note
        sub_rlm_msg = captured_messages[2]
        user_content = sub_rlm_msg[-1]["content"]
        assert "Context Note" in user_content
        assert "Chapter 1" in user_content


# =====================================================================
# RLM: mode=both
# =====================================================================


class TestCEModeBoth:
    """Tests for context_engineer_mode=both."""

    def test_both_mode_pre_loop_and_per_query(self) -> None:
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # CE pre-loop
                return "BRIEF: Research paper on methods."
            if call_count["n"] == 2:
                # Root LM
                return (
                    '```python\nresult = llm_query(CONTEXT[:200], "Summarize")\nprint(result)\n```'
                )
            if call_count["n"] == 3:
                # CE per-query
                return "NOTE: From intro section."
            if call_count["n"] == 4:
                # Sub-RLM
                return "Summary"
            return '```python\nFINAL("result")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="both",
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        # 1 pre-loop + 1 per-query = 2 CE calls
        assert result.stats.context_engineer_calls == 2


# =====================================================================
# RLM: CE helper methods
# =====================================================================


class TestCEHelperMethods:
    """Tests for _ce_enabled_* helper methods."""

    def test_ce_enabled_pre_loop(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        for mode, expected in [
            ("off", False),
            ("pre_loop", True),
            ("per_query", False),
            ("both", True),
        ]:
            rlm = RLM(backend=backend, model="m", context_engineer_mode=mode)
            assert rlm._ce_enabled_pre_loop() is expected, f"Failed for mode={mode}"

    def test_ce_enabled_per_query(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        for mode, expected in [
            ("off", False),
            ("pre_loop", False),
            ("per_query", True),
            ("both", True),
        ]:
            rlm = RLM(backend=backend, model="m", context_engineer_mode=mode)
            assert rlm._ce_enabled_per_query() is expected, f"Failed for mode={mode}"


# =====================================================================
# RLM: custom CE prompts
# =====================================================================


class TestCECustomPrompts:
    """Tests for custom context-engineer prompts."""

    def test_custom_pre_loop_prompt_used(self) -> None:
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            captured_messages.append(messages)
            if call_count["n"] == 1:
                return "Brief"
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
            context_engineer_pre_loop_prompt="CUSTOM PRE-LOOP PROMPT",
        )
        rlm.completion(SAMPLE_TEXT, "query")
        # CE pre-loop call (call #1) system message should be custom
        ce_msg = captured_messages[0]
        assert ce_msg[0]["content"] == "CUSTOM PRE-LOOP PROMPT"

    def test_custom_per_query_prompt_used(self) -> None:
        captured_messages: list[list[dict[str, str]]] = []
        call_count = {"n": 0}

        def _cb(messages: list[dict[str, str]], model: str) -> str:
            call_count["n"] += 1
            captured_messages.append(messages)
            if call_count["n"] == 1:
                return '```python\nresult = llm_query(CONTEXT[:100], "task")\nprint(result)\n```'
            if call_count["n"] == 2:
                # CE per-query
                return "Note"
            if call_count["n"] == 3:
                # Sub-RLM
                return "Response"
            return '```python\nFINAL("done")\n```'

        backend = CallbackBackend(_cb)
        rlm = RLM(
            backend=backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="per_query",
            context_engineer_per_query_prompt="CUSTOM PER-QUERY PROMPT",
        )
        rlm.completion(SAMPLE_TEXT, "query")
        # CE per-query call (call #2) system message should be custom
        ce_msg = captured_messages[1]
        assert ce_msg[0]["content"] == "CUSTOM PER-QUERY PROMPT"


# =====================================================================
# RLM: separate CE backend
# =====================================================================


class TestCESeparateBackend:
    """Tests for using a separate backend for context-engineer calls."""

    def test_separate_ce_backend(self) -> None:
        ce_call_count = {"n": 0}
        main_call_count = {"n": 0}

        def _ce_cb(messages: list[dict[str, str]], model: str) -> str:
            ce_call_count["n"] += 1
            return "CE brief from separate backend"

        def _main_cb(messages: list[dict[str, str]], model: str) -> str:
            main_call_count["n"] += 1
            return '```python\nFINAL("done")\n```'

        ce_backend = CallbackBackend(_ce_cb)
        main_backend = CallbackBackend(_main_cb)
        rlm = RLM(
            backend=main_backend,
            model="test-model",
            max_iterations=5,
            context_engineer_mode="pre_loop",
            context_engineer_backend=ce_backend,
        )
        result = rlm.completion(SAMPLE_TEXT, "query")
        assert result.success is True
        assert ce_call_count["n"] == 1
        assert main_call_count["n"] == 1


# =====================================================================
# RLM: CE model defaults
# =====================================================================


class TestCEModelDefaults:
    """Tests for context-engineer model defaulting behavior."""

    def test_ce_model_defaults_to_sub_rlm_model(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        rlm = RLM(
            backend=backend,
            model="root-model",
            sub_rlm_model="sub-model",
        )
        assert rlm.context_engineer_model == "sub-model"

    def test_ce_model_defaults_to_root_model_when_no_sub_rlm(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        rlm = RLM(backend=backend, model="root-model")
        assert rlm.context_engineer_model == "root-model"

    def test_ce_model_explicit(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        rlm = RLM(
            backend=backend,
            model="root-model",
            context_engineer_model="ce-model",
        )
        assert rlm.context_engineer_model == "ce-model"

    def test_ce_backend_defaults_to_root_backend(self) -> None:
        responses = ['```python\nFINAL("x")\n```']
        backend = make_deterministic_callback(responses)
        rlm = RLM(backend=backend, model="root-model")
        assert rlm.context_engineer_backend is backend
