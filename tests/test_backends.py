"""Unit tests for rlm.backends module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from rlm.backends import (
    ANTHROPIC_TOOL_DEFINITION,
    ANTHROPIC_TOOL_NAME,
    AnthropicBackend,
    CallbackBackend,
    ClaudeCLIBackend,
    CompletionResult,
    OpenAICompatibleBackend,
    StructuredResponse,
    TokenUsage,
    _validate_structured_fields,
)

# -----------------------------------------------------------------------
# TokenUsage dataclass
# -----------------------------------------------------------------------


class TestTokenUsage:
    def test_default_values(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_custom_values(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


# -----------------------------------------------------------------------
# CompletionResult dataclass
# -----------------------------------------------------------------------


class TestCompletionResult:
    def test_initialization(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=20)
        result = CompletionResult(text="hello", usage=usage)
        assert result.text == "hello"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20


# -----------------------------------------------------------------------
# CallbackBackend
# -----------------------------------------------------------------------


class TestCallbackBackend:
    def test_completion_calls_callback_with_correct_args(self) -> None:
        cb = MagicMock(return_value="response text")
        backend = CallbackBackend(cb)
        messages = [{"role": "user", "content": "hi"}]
        backend.completion(messages, "test-model")
        cb.assert_called_once_with(messages, "test-model")

    def test_completion_returns_completion_result(self) -> None:
        cb = MagicMock(return_value="hello world")
        backend = CallbackBackend(cb)
        result = backend.completion([{"role": "user", "content": "hi"}], "m")
        assert isinstance(result, CompletionResult)
        assert result.text == "hello world"

    def test_completion_returns_zero_token_usage(self) -> None:
        backend = CallbackBackend(lambda msgs, model: "ok")
        result = backend.completion([], "m")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_acompletion_works(self) -> None:
        backend = CallbackBackend(lambda msgs, model: "async ok")
        result = asyncio.run(backend.acompletion([{"role": "user", "content": "hi"}], "m"))
        assert result.text == "async ok"
        assert result.usage.input_tokens == 0

    def test_callback_exception_propagates(self) -> None:
        def bad_cb(msgs: list[dict[str, str]], model: str) -> str:
            raise ValueError("boom")

        backend = CallbackBackend(bad_cb)
        with pytest.raises(ValueError, match="boom"):
            backend.completion([], "m")


# -----------------------------------------------------------------------
# AnthropicBackend._split_messages
# -----------------------------------------------------------------------


class TestAnthropicSplitMessages:
    def test_system_message_extracted(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, chat = AnthropicBackend._split_messages(messages)
        assert system == "You are helpful."
        assert chat == [{"role": "user", "content": "Hello"}]

    def test_chat_messages_preserved(self) -> None:
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        system, chat = AnthropicBackend._split_messages(messages)
        assert system is None
        assert len(chat) == 3
        assert chat[0] == {"role": "user", "content": "A"}
        assert chat[1] == {"role": "assistant", "content": "B"}
        assert chat[2] == {"role": "user", "content": "C"}

    def test_no_system_message_returns_none(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        system, chat = AnthropicBackend._split_messages(messages)
        assert system is None

    def test_multiple_messages_mixed_roles(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        system, chat = AnthropicBackend._split_messages(messages)
        assert system == "sys"
        assert len(chat) == 3
        assert all(m["role"] != "system" for m in chat)

    def test_empty_messages_list(self) -> None:
        system, chat = AnthropicBackend._split_messages([])
        assert system is None
        assert chat == []


# -----------------------------------------------------------------------
# AnthropicBackend.completion (mocked client)
# -----------------------------------------------------------------------


class TestAnthropicBackendCompletion:
    def _make_backend(self) -> AnthropicBackend:
        """Create an AnthropicBackend with a mocked client."""
        with (
            patch("rlm.backends.anthropic", create=True),
            patch.object(AnthropicBackend, "__init__", lambda self, **kw: None),
        ):
            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend.client = MagicMock()
            backend._async_client = None
            return backend

    def _mock_response(
        self,
        text: str = "response",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> MagicMock:
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = text
        response.content = [content_block]
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    def test_completion_returns_text(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response(text="hello")
        result = backend.completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.text == "hello"

    def test_completion_returns_token_usage(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response(
            input_tokens=42, output_tokens=17
        )
        result = backend.completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    def test_system_message_passed_as_system_param(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ]
        backend.completion(messages, "claude-test")
        call_kwargs = backend.client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "Be helpful"
        # System message should NOT be in the messages list
        for msg in call_kwargs.kwargs["messages"]:
            assert msg["role"] != "system"

    def test_no_system_message_omits_system_param(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        messages = [{"role": "user", "content": "hi"}]
        backend.completion(messages, "claude-test")
        call_kwargs = backend.client.messages.create.call_args
        assert "system" not in call_kwargs.kwargs

    def test_default_max_tokens(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.completion([{"role": "user", "content": "hi"}], "model")
        call_kwargs = backend.client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    def test_temperature_passthrough(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.completion([{"role": "user", "content": "hi"}], "model", temperature=0.7)
        call_kwargs = backend.client.messages.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7


# -----------------------------------------------------------------------
# OpenAICompatibleBackend.completion (mocked client)
# -----------------------------------------------------------------------


class TestOpenAICompatibleBackendCompletion:
    def _make_backend(self) -> OpenAICompatibleBackend:
        """Create an OpenAICompatibleBackend with a mocked client."""
        with patch.object(OpenAICompatibleBackend, "__init__", lambda self, **kw: None):
            backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
            backend.client = MagicMock()
            backend.base_url = "http://localhost:11434/v1"
            backend._async_client = None
            return backend

    def _mock_response(
        self,
        text: str = "response",
        prompt_tokens: int | None = 10,
        completion_tokens: int | None = 5,
        has_usage: bool = True,
    ) -> MagicMock:
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        response.choices = [choice]
        if has_usage:
            response.usage.prompt_tokens = prompt_tokens
            response.usage.completion_tokens = completion_tokens
        else:
            response.usage = None
        return response

    def test_completion_returns_text(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.return_value = self._mock_response(
            text="ollama says hi"
        )
        result = backend.completion([{"role": "user", "content": "hi"}], "llama3")
        assert result.text == "ollama says hi"

    def test_completion_returns_token_usage(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.return_value = self._mock_response(
            prompt_tokens=30, completion_tokens=15
        )
        result = backend.completion([{"role": "user", "content": "hi"}], "llama3")
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 15

    def test_usage_none_returns_zero_usage(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.return_value = self._mock_response(has_usage=False)
        result = backend.completion([{"role": "user", "content": "hi"}], "llama3")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_messages_passed_as_is(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.return_value = self._mock_response()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        backend.completion(messages, "llama3")
        call_kwargs = backend.client.chat.completions.create.call_args
        # System messages are kept inline for OpenAI-compatible APIs
        assert call_kwargs.kwargs["messages"] is messages

    def test_none_content_returns_empty_string(self) -> None:
        backend = self._make_backend()
        response = self._mock_response()
        response.choices[0].message.content = None
        backend.client.chat.completions.create.return_value = response
        result = backend.completion([{"role": "user", "content": "hi"}], "m")
        assert result.text == ""


# -----------------------------------------------------------------------
# ClaudeCLIBackend.__init__
# -----------------------------------------------------------------------


class TestClaudeCLIBackendInit:
    def test_file_not_found_when_not_on_path(self) -> None:
        with (
            patch("rlm.backends.shutil.which", return_value=None),
            pytest.raises(FileNotFoundError, match="not found on PATH"),
        ):
            ClaudeCLIBackend(claude_cmd="claude")

    def test_init_succeeds_when_on_path(self) -> None:
        with patch("rlm.backends.shutil.which", return_value="/usr/bin/claude"):
            backend = ClaudeCLIBackend(claude_cmd="claude")
            assert backend._cmd == "/usr/bin/claude"
            assert backend._max_turns == 1


# -----------------------------------------------------------------------
# ClaudeCLIBackend._format_conversation
# -----------------------------------------------------------------------


class TestClaudeCLIFormatConversation:
    def test_system_prompt_extracted(self) -> None:
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, user_prompt = ClaudeCLIBackend._format_conversation(messages)
        assert system_prompt == "Be concise."
        assert "[user]\nHello" in user_prompt

    def test_multi_turn_serialized(self) -> None:
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        system_prompt, user_prompt = ClaudeCLIBackend._format_conversation(messages)
        assert system_prompt is None
        assert "[user]\nQ1" in user_prompt
        assert "[assistant]\nA1" in user_prompt
        assert "[user]\nQ2" in user_prompt

    def test_no_system_message_returns_none(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        system_prompt, _ = ClaudeCLIBackend._format_conversation(messages)
        assert system_prompt is None

    def test_only_user_messages(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        system_prompt, user_prompt = ClaudeCLIBackend._format_conversation(messages)
        assert system_prompt is None
        assert "[user]\nfirst" in user_prompt
        assert "[user]\nsecond" in user_prompt


# -----------------------------------------------------------------------
# ClaudeCLIBackend._run_claude (mocked subprocess)
# -----------------------------------------------------------------------


class TestClaudeCLIRunClaude:
    def _make_backend(self) -> ClaudeCLIBackend:
        with patch("rlm.backends.shutil.which", return_value="/usr/bin/claude"):
            return ClaudeCLIBackend()

    def test_successful_json_response(self) -> None:
        backend = self._make_backend()
        json_output = json.dumps({"result": "answer", "input_tokens": 100, "output_tokens": 50})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc):
            result = backend._run_claude(None, "prompt", "sonnet")

        assert result.text == "answer"
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

    def test_token_usage_extracted(self) -> None:
        backend = self._make_backend()
        json_output = json.dumps({"result": "ok", "input_tokens": 200, "output_tokens": 80})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc):
            result = backend._run_claude(None, "prompt", "m")

        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 80

    def test_nonzero_return_code_raises(self) -> None:
        backend = self._make_backend()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "error message"

        with (
            patch("rlm.backends.subprocess.run", return_value=mock_proc),
            pytest.raises(RuntimeError, match="claude exited with code 1"),
        ):
            backend._run_claude(None, "prompt", "m")

    def test_non_json_response_falls_back_to_raw_text(self) -> None:
        backend = self._make_backend()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "plain text response"
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc):
            result = backend._run_claude(None, "prompt", "m")

        assert result.text == "plain text response"
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_system_prompt_passed_as_flag(self) -> None:
        backend = self._make_backend()
        json_output = json.dumps({"result": "ok"})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc) as mock_run:
            backend._run_claude("Be concise", "prompt", "m")

        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "Be concise"

    def test_no_system_prompt_omits_flag(self) -> None:
        backend = self._make_backend()
        json_output = json.dumps({"result": "ok"})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc) as mock_run:
            backend._run_claude(None, "prompt", "m")

        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" not in cmd

    def test_json_without_token_fields_returns_zero_usage(self) -> None:
        backend = self._make_backend()
        json_output = json.dumps({"result": "ok"})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        with patch("rlm.backends.subprocess.run", return_value=mock_proc):
            result = backend._run_claude(None, "prompt", "m")

        assert result.text == "ok"
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# -----------------------------------------------------------------------
# ClaudeCLIBackend.completion (full flow)
# -----------------------------------------------------------------------


class TestClaudeCLIBackendCompletion:
    def test_full_flow(self) -> None:
        with patch("rlm.backends.shutil.which", return_value="/usr/bin/claude"):
            backend = ClaudeCLIBackend()

        json_output = json.dumps({"result": "the answer", "input_tokens": 50, "output_tokens": 25})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json_output
        mock_proc.stderr = ""

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        with patch("rlm.backends.subprocess.run", return_value=mock_proc) as mock_run:
            result = backend.completion(messages, "sonnet")

        assert result.text == "the answer"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 25

        cmd = mock_run.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "You are helpful." in cmd


# -----------------------------------------------------------------------
# LLMBackend.structured_completion (ABC default)
# -----------------------------------------------------------------------


class TestLLMBackendStructuredCompletion:
    def test_default_raises_not_implemented(self) -> None:
        """ABC default structured_completion() raises NotImplementedError."""
        backend = CallbackBackend(lambda _msgs, _model: "ok")
        with pytest.raises(NotImplementedError, match="does not support structured output"):
            backend.structured_completion([{"role": "user", "content": "hi"}], "m")

    def test_callback_backend_does_not_support_structured_output(self) -> None:
        backend = CallbackBackend(lambda _msgs, _model: "ok")
        assert backend.supports_structured_output is False

    def test_anthropic_backend_supports_structured_output(self) -> None:
        with (
            patch("rlm.backends.anthropic", create=True),
            patch.object(AnthropicBackend, "__init__", lambda _self, **_kw: None),
        ):
            backend = AnthropicBackend.__new__(AnthropicBackend)
            assert backend.supports_structured_output is True


# -----------------------------------------------------------------------
# OpenAICompatibleBackend.supports_structured_output
# -----------------------------------------------------------------------


class TestOpenAICompatibleSupportsStructured:
    def test_returns_true(self) -> None:
        with patch.object(OpenAICompatibleBackend, "__init__", lambda _self, **_kw: None):
            backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
            assert backend.supports_structured_output is True


# -----------------------------------------------------------------------
# OpenAICompatibleBackend._parse_structured_response
# -----------------------------------------------------------------------


class TestParseStructuredResponse:
    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "reasoning": "I need to search the document.",
                "code": "print(len(CONTEXT))",
                "is_final": False,
                "final_answer": None,
            }
        )
        result = OpenAICompatibleBackend._parse_structured_response(raw)
        assert result is not None
        assert result.reasoning == "I need to search the document."
        assert result.code == "print(len(CONTEXT))"
        assert result.is_final is False
        assert result.final_answer is None

    def test_final_answer(self) -> None:
        raw = json.dumps(
            {
                "reasoning": "Done.",
                "code": None,
                "is_final": True,
                "final_answer": "The answer is 42.",
            }
        )
        result = OpenAICompatibleBackend._parse_structured_response(raw)
        assert result is not None
        assert result.is_final is True
        assert result.final_answer == "The answer is 42."
        assert result.code is None

    def test_malformed_json_returns_none(self) -> None:
        assert OpenAICompatibleBackend._parse_structured_response("not json{") is None

    def test_missing_required_field_returns_none(self) -> None:
        raw = json.dumps({"reasoning": "ok", "code": None})
        assert OpenAICompatibleBackend._parse_structured_response(raw) is None

    def test_non_object_json_returns_none(self) -> None:
        assert OpenAICompatibleBackend._parse_structured_response('"just a string"') is None

    def test_none_input_returns_none(self) -> None:
        assert OpenAICompatibleBackend._parse_structured_response(None) is None  # type: ignore[arg-type]

    def test_wrong_type_reasoning_returns_none(self) -> None:
        raw = json.dumps({"reasoning": 123, "code": None, "is_final": True, "final_answer": None})
        assert OpenAICompatibleBackend._parse_structured_response(raw) is None

    def test_wrong_type_is_final_string_returns_none(self) -> None:
        """Ensures 'false' string is not coerced to True via bool()."""
        raw = json.dumps(
            {
                "reasoning": "r",
                "code": None,
                "is_final": "false",
                "final_answer": None,
            }
        )
        assert OpenAICompatibleBackend._parse_structured_response(raw) is None

    def test_wrong_type_code_returns_none(self) -> None:
        raw = json.dumps(
            {
                "reasoning": "r",
                "code": 42,
                "is_final": False,
                "final_answer": None,
            }
        )
        assert OpenAICompatibleBackend._parse_structured_response(raw) is None

    def test_wrong_type_final_answer_returns_none(self) -> None:
        raw = json.dumps(
            {
                "reasoning": "r",
                "code": None,
                "is_final": True,
                "final_answer": ["not", "a", "string"],
            }
        )
        assert OpenAICompatibleBackend._parse_structured_response(raw) is None


# -----------------------------------------------------------------------
# OpenAICompatibleBackend.structured_completion (mocked client)
# -----------------------------------------------------------------------


class TestOpenAICompatibleStructuredCompletion:
    def _make_backend(self) -> OpenAICompatibleBackend:
        """Create an OpenAICompatibleBackend with a mocked client."""
        with patch.object(OpenAICompatibleBackend, "__init__", lambda _self, **_kw: None):
            backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
            backend.client = MagicMock()
            backend.base_url = "http://localhost:11434/v1"
            backend._async_client = None
            return backend

    def _mock_response(
        self,
        text: str = "response",
        prompt_tokens: int | None = 10,
        completion_tokens: int | None = 5,
        has_usage: bool = True,
    ) -> MagicMock:
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        response.choices = [choice]
        if has_usage:
            response.usage.prompt_tokens = prompt_tokens
            response.usage.completion_tokens = completion_tokens
        else:
            response.usage = None
        return response

    def test_happy_path_returns_structured(self) -> None:
        backend = self._make_backend()
        json_text = json.dumps(
            {
                "reasoning": "Searching...",
                "code": "print(CONTEXT[:100])",
                "is_final": False,
                "final_answer": None,
            }
        )
        backend.client.chat.completions.create.return_value = self._mock_response(text=json_text)
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "gpt-4o")

        assert result.structured is not None
        assert isinstance(result.structured, StructuredResponse)
        assert result.structured.reasoning == "Searching..."
        assert result.structured.code == "print(CONTEXT[:100])"
        assert result.structured.is_final is False
        assert result.text == json_text

    def test_response_format_passed_to_api(self) -> None:
        backend = self._make_backend()
        json_text = json.dumps(
            {
                "reasoning": "ok",
                "code": None,
                "is_final": True,
                "final_answer": "done",
            }
        )
        backend.client.chat.completions.create.return_value = self._mock_response(text=json_text)
        backend.structured_completion([{"role": "user", "content": "hi"}], "gpt-4o")

        call_kwargs = backend.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    def test_malformed_json_returns_none_structured(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.return_value = self._mock_response(
            text="This is not JSON at all"
        )
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "m")

        assert result.structured is None
        assert result.text == "This is not JSON at all"

    def test_token_usage_populated(self) -> None:
        backend = self._make_backend()
        json_text = json.dumps(
            {
                "reasoning": "r",
                "code": None,
                "is_final": True,
                "final_answer": "a",
            }
        )
        backend.client.chat.completions.create.return_value = self._mock_response(
            text=json_text, prompt_tokens=42, completion_tokens=17
        )
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "m")

        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    def test_temperature_and_max_tokens_passthrough(self) -> None:
        backend = self._make_backend()
        json_text = json.dumps(
            {
                "reasoning": "r",
                "code": None,
                "is_final": True,
                "final_answer": "a",
            }
        )
        backend.client.chat.completions.create.return_value = self._mock_response(text=json_text)
        backend.structured_completion(
            [{"role": "user", "content": "hi"}], "m", temperature=0.5, max_tokens=1000
        )

        call_kwargs = backend.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 1000

    def test_api_error_propagates(self) -> None:
        backend = self._make_backend()
        backend.client.chat.completions.create.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            backend.structured_completion([{"role": "user", "content": "hi"}], "m")


# -----------------------------------------------------------------------
# _validate_structured_fields (module-level helper)
# -----------------------------------------------------------------------


class TestValidateStructuredFields:
    def test_happy_path(self) -> None:
        data = {
            "reasoning": "Searching the document.",
            "code": "print(len(CONTEXT))",
            "is_final": False,
            "final_answer": None,
        }
        result = _validate_structured_fields(data)
        assert result is not None
        assert result.reasoning == "Searching the document."
        assert result.code == "print(len(CONTEXT))"
        assert result.is_final is False
        assert result.final_answer is None

    def test_final_answer_scenario(self) -> None:
        data = {
            "reasoning": "Done.",
            "code": None,
            "is_final": True,
            "final_answer": "The answer is 42.",
        }
        result = _validate_structured_fields(data)
        assert result is not None
        assert result.is_final is True
        assert result.final_answer == "The answer is 42."

    def test_missing_fields_returns_none(self) -> None:
        assert _validate_structured_fields({"reasoning": "ok"}) is None

    def test_wrong_type_reasoning_returns_none(self) -> None:
        data = {"reasoning": 123, "code": None, "is_final": True, "final_answer": None}
        assert _validate_structured_fields(data) is None

    def test_wrong_type_is_final_returns_none(self) -> None:
        data = {"reasoning": "r", "code": None, "is_final": "false", "final_answer": None}
        assert _validate_structured_fields(data) is None

    def test_wrong_type_code_returns_none(self) -> None:
        data = {"reasoning": "r", "code": 42, "is_final": False, "final_answer": None}
        assert _validate_structured_fields(data) is None

    def test_wrong_type_final_answer_returns_none(self) -> None:
        data = {"reasoning": "r", "code": None, "is_final": True, "final_answer": ["list"]}
        assert _validate_structured_fields(data) is None


# -----------------------------------------------------------------------
# AnthropicBackend._extract_tool_use_input
# -----------------------------------------------------------------------


class TestAnthropicExtractToolUseInput:
    @staticmethod
    def _make_tool_use_block(
        name: str = ANTHROPIC_TOOL_NAME,
        input_data: dict[str, object] | None = None,
    ) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.input = input_data or {
            "reasoning": "r",
            "code": None,
            "is_final": True,
            "final_answer": "a",
        }
        return block

    @staticmethod
    def _make_text_block(text: str = "some text") -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def test_tool_use_found(self) -> None:
        content = [self._make_tool_use_block()]
        tool_input, text = AnthropicBackend._extract_tool_use_input(content)
        assert tool_input is not None
        assert tool_input["reasoning"] == "r"
        assert text == ""

    def test_no_tool_use_block(self) -> None:
        content = [self._make_text_block("hello")]
        tool_input, text = AnthropicBackend._extract_tool_use_input(content)
        assert tool_input is None
        assert text == "hello"

    def test_wrong_tool_name_ignored(self) -> None:
        content = [self._make_tool_use_block(name="other_tool")]
        tool_input, text = AnthropicBackend._extract_tool_use_input(content)
        assert tool_input is None

    def test_mixed_blocks(self) -> None:
        content = [
            self._make_text_block("prefix "),
            self._make_tool_use_block(),
            self._make_text_block("suffix"),
        ]
        tool_input, text = AnthropicBackend._extract_tool_use_input(content)
        assert tool_input is not None
        assert text == "prefix suffix"

    def test_empty_content(self) -> None:
        tool_input, text = AnthropicBackend._extract_tool_use_input([])
        assert tool_input is None
        assert text == ""


# -----------------------------------------------------------------------
# AnthropicBackend.structured_completion (mocked client)
# -----------------------------------------------------------------------


class TestAnthropicStructuredCompletion:
    def _make_backend(self) -> AnthropicBackend:
        """Create an AnthropicBackend with a mocked client."""
        with (
            patch("rlm.backends.anthropic", create=True),
            patch.object(AnthropicBackend, "__init__", lambda self, **kw: None),
        ):
            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend.client = MagicMock()
            backend._async_client = None
            return backend

    def _mock_response(
        self,
        content: list[MagicMock] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> MagicMock:
        response = MagicMock()
        if content is not None:
            response.content = content
        else:
            # Default: a tool_use block with valid input
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = ANTHROPIC_TOOL_NAME
            tool_block.input = {
                "reasoning": "Searching...",
                "code": "print(CONTEXT[:100])",
                "is_final": False,
                "final_answer": None,
            }
            response.content = [tool_block]
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    def test_happy_path_returns_structured(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.structured is not None
        assert isinstance(result.structured, StructuredResponse)
        assert result.structured.reasoning == "Searching..."
        assert result.structured.code == "print(CONTEXT[:100])"
        assert result.structured.is_final is False

    def test_tools_and_tool_choice_passed_to_api(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.structured_completion([{"role": "user", "content": "hi"}], "claude-test")

        call_kwargs = backend.client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == [ANTHROPIC_TOOL_DEFINITION]
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": ANTHROPIC_TOOL_NAME,
        }

    def test_system_message_splitting(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
        ]
        backend.structured_completion(messages, "claude-test")
        call_kwargs = backend.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_malformed_tool_input_returns_none_structured(self) -> None:
        backend = self._make_backend()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = ANTHROPIC_TOOL_NAME
        tool_block.input = {"reasoning": 123}  # wrong type, missing fields
        backend.client.messages.create.return_value = self._mock_response(content=[tool_block])
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.structured is None
        # text should be the JSON serialisation of the tool input
        assert result.text == json.dumps({"reasoning": 123})

    def test_no_tool_use_block_returns_text(self) -> None:
        backend = self._make_backend()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I cannot use tools right now."
        backend.client.messages.create.return_value = self._mock_response(content=[text_block])
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.structured is None
        assert result.text == "I cannot use tools right now."

    def test_token_usage_populated(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response(
            input_tokens=42, output_tokens=17
        )
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "claude-test")
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    def test_temperature_passthrough(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.structured_completion([{"role": "user", "content": "hi"}], "m", temperature=0.5)
        call_kwargs = backend.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_default_max_tokens(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.structured_completion([{"role": "user", "content": "hi"}], "m")
        call_kwargs = backend.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_custom_max_tokens(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.return_value = self._mock_response()
        backend.structured_completion([{"role": "user", "content": "hi"}], "m", max_tokens=1000)
        call_kwargs = backend.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1000

    def test_api_error_propagates(self) -> None:
        backend = self._make_backend()
        backend.client.messages.create.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            backend.structured_completion([{"role": "user", "content": "hi"}], "m")

    def test_final_answer_scenario(self) -> None:
        backend = self._make_backend()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = ANTHROPIC_TOOL_NAME
        tool_block.input = {
            "reasoning": "Done.",
            "code": None,
            "is_final": True,
            "final_answer": "The answer is 42.",
        }
        backend.client.messages.create.return_value = self._mock_response(content=[tool_block])
        result = backend.structured_completion([{"role": "user", "content": "hi"}], "m")
        assert result.structured is not None
        assert result.structured.is_final is True
        assert result.structured.final_answer == "The answer is 42."
        assert result.structured.code is None
