"""Unit tests for rlm.prompts module."""

from __future__ import annotations

from rlm.prompts import get_system_prompt, get_user_prompt

# ---------------------------------------------------------------------------
# get_system_prompt()
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    """Tests for get_system_prompt()."""

    def test_returns_full_prompt_by_default(self) -> None:
        prompt = get_system_prompt()
        # Full prompt is significantly longer
        assert len(prompt) > 1000

    def test_returns_compact_prompt_when_compact_true(self) -> None:
        prompt = get_system_prompt(compact=True)
        assert len(prompt) > 0

    def test_full_prompt_longer_than_compact(self) -> None:
        full = get_system_prompt(compact=False)
        compact = get_system_prompt(compact=True)
        assert len(full) > len(compact)

    def test_full_prompt_contains_key_instructions(self) -> None:
        prompt = get_system_prompt(compact=False)
        assert "CONTEXT" in prompt
        assert "FINAL" in prompt
        assert "llm_query" in prompt
        assert "code" in prompt

    def test_compact_prompt_contains_key_instructions(self) -> None:
        prompt = get_system_prompt(compact=True)
        assert "CONTEXT" in prompt
        assert "FINAL" in prompt
        assert "llm_query" in prompt


# ---------------------------------------------------------------------------
# get_user_prompt()
# ---------------------------------------------------------------------------


class TestGetUserPrompt:
    """Tests for get_user_prompt()."""

    def test_includes_query_text(self) -> None:
        prompt = get_user_prompt("What are the main themes?")
        assert "What are the main themes?" in prompt

    def test_contains_final_instruction(self) -> None:
        prompt = get_user_prompt("test query")
        assert "FINAL" in prompt

    def test_contains_python_code_instruction(self) -> None:
        prompt = get_user_prompt("test query")
        assert "Python code" in prompt
