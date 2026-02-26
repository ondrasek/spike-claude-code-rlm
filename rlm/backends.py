"""LLM backend implementations for RLM.

Supports multiple LLM providers:
- AnthropicBackend: Direct Anthropic API
- OpenAICompatibleBackend: OpenAI-compatible APIs (Ollama, vLLM, etc.)
- ClaudeCLIBackend: Claude Code CLI (claude -p)
- CallbackBackend: Internal — used by the test suite only
"""

import json
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StructuredResponse:
    """Parsed structured output from an LLM backend."""

    reasoning: str
    code: str | None = None
    is_final: bool = False
    final_answer: str | None = None


STRUCTURED_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "code": {"type": ["string", "null"]},
        "is_final": {"type": "boolean"},
        "final_answer": {"type": ["string", "null"]},
    },
    "required": ["reasoning", "code", "is_final", "final_answer"],
    "additionalProperties": False,
}


@dataclass
class CompletionResult:
    """Result from a backend completion call, including token usage."""

    text: str
    usage: TokenUsage
    structured: StructuredResponse | None = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    Implementations must handle system messages according to their provider's
    API requirements.  The orchestrator includes a ``{"role": "system", ...}``
    message as the first element of the ``messages`` list.  Backends that need
    to separate system messages (e.g. Anthropic) should extract them before
    forwarding to the API.  Backends whose APIs accept system messages inline
    (e.g. OpenAI-compatible) can pass the list as-is.
    """

    @property
    def supports_structured_output(self) -> bool:
        """Whether this backend supports structured output responses."""
        return False

    def structured_completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate a structured JSON completion.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with 'role' and 'content' keys.
        model : str
            Model identifier.
        **kwargs
            Additional provider-specific parameters.

        Returns
        -------
        CompletionResult
            Result with ``structured`` field populated on success.

        Raises
        ------
        NotImplementedError
            If the backend has not implemented structured output.
        """
        raise NotImplementedError("Backend does not support structured output")

    @abstractmethod
    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate completion from messages.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with 'role' and 'content' keys.
            May include a ``{"role": "system", ...}`` entry.
        model : str
            Model identifier.
        **kwargs
            Additional provider-specific parameters (e.g. ``max_tokens``,
            ``temperature``).

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """

    @abstractmethod
    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Async version of completion.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with 'role' and 'content' keys.
        model : str
            Model identifier.
        **kwargs
            Additional provider-specific parameters.

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """


class AnthropicBackend(LLMBackend):
    """Backend for Anthropic's Claude models."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Anthropic backend.

        Parameters
        ----------
        api_key : str | None
            Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._async_client: Any = None

    def _get_async_client(self) -> Any:
        """Return a reusable async Anthropic client."""
        if self._async_client is None:
            import anthropic

            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.client.api_key,
            )
        return self._async_client

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Separate system messages from chat messages.

        Anthropic's API requires system content to be passed via a dedicated
        ``system`` parameter rather than as a message with role ``system``.
        """
        system_message: str | None = None
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
        return system_message, chat_messages

    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate completion using Anthropic API.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Claude model identifier.
        **kwargs
            Additional parameters (max_tokens, temperature, etc.).

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        system_message, chat_messages = self._split_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": chat_messages,
        }

        if system_message:
            params["system"] = system_message

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        response = self.client.messages.create(**params)
        text: str = response.content[0].text
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return CompletionResult(text=text, usage=usage)

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Async completion using Anthropic API.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Claude model identifier.
        **kwargs
            Additional parameters.

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        system_message, chat_messages = self._split_messages(messages)

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": chat_messages,
        }

        if system_message:
            params["system"] = system_message

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        async_client = self._get_async_client()
        response = await async_client.messages.create(**params)
        text: str = response.content[0].text
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return CompletionResult(text=text, usage=usage)


class OpenAICompatibleBackend(LLMBackend):
    """Backend for OpenAI-compatible APIs (Ollama, vLLM, LM Studio, etc.)."""

    @property
    def supports_structured_output(self) -> bool:
        """OpenAI-compatible APIs support JSON mode."""
        return True

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        """Initialize OpenAI-compatible backend.

        Parameters
        ----------
        base_url : str
            Base URL for the API endpoint.
        api_key : str
            API key (many local servers don't require a real key).
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self._async_client: Any = None

    def _get_async_client(self) -> Any:
        """Return a reusable async OpenAI client."""
        if self._async_client is None:
            import openai

            self._async_client = openai.AsyncOpenAI(
                base_url=self.base_url, api_key=self.client.api_key
            )
        return self._async_client

    @staticmethod
    def _parse_structured_response(raw: str) -> StructuredResponse | None:
        """Parse a JSON string into a ``StructuredResponse``.

        Returns ``None`` when the JSON is malformed or missing required fields.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Structured output: failed to parse JSON")
            return None

        if not isinstance(data, dict):
            logger.debug("Structured output: response is not a JSON object")
            return None

        required = {"reasoning", "code", "is_final", "final_answer"}
        if not required.issubset(data):
            logger.debug("Structured output: missing required fields %s", required - data.keys())
            return None

        reasoning = data["reasoning"]
        code = data["code"]
        is_final = data["is_final"]
        final_answer = data["final_answer"]

        if not isinstance(reasoning, str):
            logger.debug("Structured output: 'reasoning' must be a string")
            return None
        if code is not None and not isinstance(code, str):
            logger.debug("Structured output: 'code' must be a string or null")
            return None
        if not isinstance(is_final, bool):
            logger.debug("Structured output: 'is_final' must be a boolean")
            return None
        if final_answer is not None and not isinstance(final_answer, str):
            logger.debug("Structured output: 'final_answer' must be a string or null")
            return None

        return StructuredResponse(
            reasoning=reasoning,
            code=code,
            is_final=is_final,
            final_answer=final_answer,
        )

    @staticmethod
    def _build_result(
        response: Any,
        *,
        structured: StructuredResponse | None = None,
    ) -> CompletionResult:
        """Extract text, token usage, and optional structured data from a raw API response."""
        text = response.choices[0].message.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        return CompletionResult(text=text, usage=usage, structured=structured)

    def structured_completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate a structured JSON completion using ``response_format``.

        Sends the request with ``response_format={"type": "json_object"}`` and
        parses the response into a :class:`StructuredResponse`.  If parsing
        fails the result is returned with ``structured=None`` so the caller
        can fall back to regex-based extraction.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Model identifier.
        **kwargs
            Additional parameters (``temperature``, ``max_tokens``, etc.).

        Returns
        -------
        CompletionResult
            Result with ``structured`` populated on success, ``None`` on
            parse failure.
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]

        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""
        structured = self._parse_structured_response(text)
        return self._build_result(response, structured=structured)

    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate completion using OpenAI-compatible API.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Model identifier.
        **kwargs
            Additional parameters.

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]

        response = self.client.chat.completions.create(**params)
        return self._build_result(response)

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Async completion using OpenAI-compatible API.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Model identifier.
        **kwargs
            Additional parameters.

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        async_client = self._get_async_client()

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]

        response = await async_client.chat.completions.create(**params)
        return self._build_result(response)


class CallbackBackend(LLMBackend):
    """Backend using custom callback function.

    Useful for integrating with Claude Max, CLI tools, or other custom systems.
    """

    def __init__(self, callback_fn: Callable[[list[dict[str, str]], str], str]) -> None:
        """Initialize callback backend.

        Parameters
        ----------
        callback_fn : Callable
            Function that takes (messages, model) and returns response string.
        """
        self.callback_fn = callback_fn

    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate completion using callback function.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Model identifier.
        **kwargs
            Additional parameters (passed but may be ignored by callback).

        Returns
        -------
        CompletionResult
            Generated text response (token usage will be zero).
        """
        text = self.callback_fn(messages, model)
        return CompletionResult(text=text, usage=TokenUsage())

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Async completion using callback (runs synchronously).

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts.
        model : str
            Model identifier.
        **kwargs
            Additional parameters.

        Returns
        -------
        CompletionResult
            Generated text response (token usage will be zero).
        """
        text = self.callback_fn(messages, model)
        return CompletionResult(text=text, usage=TokenUsage())


class ClaudeCLIBackend(LLMBackend):
    """Backend that shells out to ``claude -p`` (print mode).

    Uses the Claude Code CLI in non-interactive mode for each LLM call.
    This is useful when you want to leverage an existing Claude Code / Claude
    Max subscription instead of providing a raw API key.

    No extra Python dependencies are required — only the ``claude`` binary on
    ``$PATH``.
    """

    def __init__(
        self,
        claude_cmd: str = "claude",
        *,
        max_turns: int = 1,
    ) -> None:
        """Initialize the Claude CLI backend.

        Parameters
        ----------
        claude_cmd : str
            Path or name of the ``claude`` binary (default: ``"claude"``).
        max_turns : int
            ``--max-turns`` passed to ``claude -p`` (default: 1).
            Each RLM backend call is a single-shot prompt, so 1 is usually
            correct.  Increase only if the CLI needs agentic follow-up turns
            to produce a response.
        """
        resolved = shutil.which(claude_cmd)
        if resolved is None:
            raise FileNotFoundError(
                f"'{claude_cmd}' not found on PATH. "
                "Install Claude Code: https://docs.anthropic.com/en/docs/claude-code"
            )
        self._cmd = resolved
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        """Split messages into a system prompt and a single user prompt.

        The Claude CLI accepts one prompt string (the last user turn) and an
        optional ``--system-prompt``.  Multi-turn history is serialised into
        the prompt with role markers so the model sees the full conversation.

        Returns
        -------
        tuple[str | None, str]
            ``(system_prompt, user_prompt)``
        """
        system_prompt: str | None = None
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_prompt = content
            elif role == "assistant":
                parts.append(f"[assistant]\n{content}")
            else:
                parts.append(f"[user]\n{content}")

        user_prompt = "\n\n".join(parts)
        return system_prompt, user_prompt

    def _run_claude(
        self,
        system_prompt: str | None,
        user_prompt: str,
        model: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute ``claude -p`` and return the result.

        Parameters
        ----------
        system_prompt : str | None
            Optional system prompt.
        user_prompt : str
            The prompt to send.
        model : str
            Model alias or full model ID.
        **kwargs
            Unused (present for signature compatibility).

        Returns
        -------
        CompletionResult
            Response text and (estimated) token usage.
        """
        cmd: list[str] = [
            self._cmd,
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--max-turns",
            str(self._max_turns),
            "--no-session-persistence",
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        cmd.append(user_prompt)

        logger.debug("Running: %s", " ".join(cmd[:6]) + " ...")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(f"claude exited with code {proc.returncode}: {stderr}")

        # --output-format json returns a JSON object with a "result" field.
        raw = proc.stdout.strip()
        usage = TokenUsage()
        try:
            data = json.loads(raw)
            text = data.get("result", raw)
            # Populate token usage if the CLI reports it.
            if data.get("input_tokens") is not None:
                usage = TokenUsage(
                    input_tokens=int(data["input_tokens"]),
                    output_tokens=int(data.get("output_tokens", 0)),
                )
        except (json.JSONDecodeError, TypeError):
            # Fall back to raw text if the output is not valid JSON.
            text = raw

        return CompletionResult(text=text, usage=usage)

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def completion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Generate completion by calling ``claude -p``.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Conversation messages.
        model : str
            Model alias (``sonnet``, ``opus``) or full model ID.
        **kwargs
            Additional parameters (currently unused).

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        system_prompt, user_prompt = self._format_conversation(messages)
        return self._run_claude(system_prompt, user_prompt, model, **kwargs)

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResult:
        """Async completion via ``claude -p`` (runs synchronously).

        Parameters
        ----------
        messages : list[dict[str, str]]
            Conversation messages.
        model : str
            Model alias or full model ID.
        **kwargs
            Additional parameters.

        Returns
        -------
        CompletionResult
            Generated text response with token usage.
        """
        system_prompt, user_prompt = self._format_conversation(messages)
        return self._run_claude(system_prompt, user_prompt, model, **kwargs)
