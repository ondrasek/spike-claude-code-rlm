"""LLM backend implementations for RLM.

Supports multiple LLM providers:
- AnthropicBackend: Direct Anthropic API
- OpenAICompatibleBackend: OpenAI-compatible APIs (Ollama, vLLM, etc.)
- CallbackBackend: Custom callback function for integration with other systems
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """Token usage statistics from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CompletionResult:
    """Result from a backend completion call, including token usage."""

    text: str
    usage: TokenUsage


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    Implementations must handle system messages according to their provider's
    API requirements.  The orchestrator includes a ``{"role": "system", ...}``
    message as the first element of the ``messages`` list.  Backends that need
    to separate system messages (e.g. Anthropic) should extract them before
    forwarding to the API.  Backends whose APIs accept system messages inline
    (e.g. OpenAI-compatible) can pass the list as-is.
    """

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
            import anthropic  # type: ignore[import-not-found]
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
            import openai  # type: ignore[import-not-found]
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
        text = response.choices[0].message.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        return CompletionResult(text=text, usage=usage)

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
        text = response.choices[0].message.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        return CompletionResult(text=text, usage=usage)


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
