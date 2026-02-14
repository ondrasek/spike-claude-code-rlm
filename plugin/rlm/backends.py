"""LLM backend implementations for RLM.

Supports multiple LLM providers:
- AnthropicBackend: Direct Anthropic API
- OpenAICompatibleBackend: OpenAI-compatible APIs (Ollama, vLLM, etc.)
- CallbackBackend: Custom callback function for integration with other systems
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Callable


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def completion(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        """Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> str:
        """Async version of completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        pass


class AnthropicBackend(LLMBackend):
    """Backend for Anthropic's Claude models."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Anthropic backend.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def completion(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        """Generate completion using Anthropic API.

        Args:
            messages: List of message dicts
            model: Claude model identifier
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            Generated text response
        """
        # Anthropic requires separating system messages
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        # Default parameters
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
        return response.content[0].text

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> str:
        """Async completion using Anthropic API.

        Args:
            messages: List of message dicts
            model: Claude model identifier
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Anthropic requires separating system messages
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": chat_messages,
        }

        if system_message:
            params["system"] = system_message

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        # Use async client
        import anthropic

        async_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY") or self.client.api_key
        )
        response = await async_client.messages.create(**params)
        return response.content[0].text


class OpenAICompatibleBackend(LLMBackend):
    """Backend for OpenAI-compatible APIs (Ollama, vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        """Initialize OpenAI-compatible backend.

        Args:
            base_url: Base URL for the API endpoint
            api_key: API key (many local servers don't require a real key)
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url

    def completion(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        """Generate completion using OpenAI-compatible API.

        Args:
            messages: List of message dicts
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            Generated text response
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
        return response.choices[0].message.content or ""

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> str:
        """Async completion using OpenAI-compatible API.

        Args:
            messages: List of message dicts
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        import openai

        async_client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.client.api_key)

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]

        response = await async_client.chat.completions.create(**params)
        return response.choices[0].message.content or ""


class CallbackBackend(LLMBackend):
    """Backend using custom callback function.

    Useful for integrating with Claude Max, CLI tools, or other custom systems.
    """

    def __init__(self, callback_fn: Callable[[list[dict[str, str]], str], str]) -> None:
        """Initialize callback backend.

        Args:
            callback_fn: Function that takes (messages, model) and returns response string
        """
        self.callback_fn = callback_fn

    def completion(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
        """Generate completion using callback function.

        Args:
            messages: List of message dicts
            model: Model identifier
            **kwargs: Additional parameters (passed but may be ignored by callback)

        Returns:
            Generated text response
        """
        return self.callback_fn(messages, model)

    async def acompletion(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> str:
        """Async completion using callback (runs synchronously).

        Args:
            messages: List of message dicts
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # For simplicity, run callback synchronously
        # Users can provide an async callback if needed
        return self.callback_fn(messages, model)
