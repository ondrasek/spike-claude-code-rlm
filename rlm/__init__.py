"""RLM - Recursive Language Model.

A Python 3.11+ implementation of the Recursive Language Model paradigm
from MIT CSAIL research (arXiv:2512.24601).

Unlike traditional RAG (Retrieval-Augmented Generation), RLM treats document
context as an external variable in a Python REPL environment. The LLM writes
Python code to inspect, search, chunk, and recursively process documents far
exceeding typical context windows.

Example:
    >>> from rlm import RLM
    >>> from rlm.backends import AnthropicBackend
    >>>
    >>> backend = AnthropicBackend()
    >>> rlm = RLM(backend, model="claude-sonnet-4-20250514", verbose=True)
    >>>
    >>> with open("document.txt") as f:
    ...     context = f.read()
    >>>
    >>> result = rlm.completion(
    ...     context=context,
    ...     query="What are the main themes?"
    ... )
    >>> print(result.answer)
"""

from .backends import (
    STRUCTURED_RESPONSE_SCHEMA,
    AnthropicBackend,
    ClaudeCLIBackend,
    CompletionResult,
    LLMBackend,
    OpenAICompatibleBackend,
    StructuredResponse,
    TokenUsage,
)
from .config import (
    ConfigError,
    DefaultsConfig,
    ResolvedRoleConfig,
    RLMConfig,
    RoleConfig,
    SettingsConfig,
    load_config,
)
from .context import CompositeContext, LazyContext, StringContext
from .repl import REPLEnv, REPLResult
from .rlm import RLM, RLMResult, RLMStats

__version__ = "0.1.0"

__all__ = [
    "RLM",
    "RLMResult",
    "RLMStats",
    "LLMBackend",
    "AnthropicBackend",
    "OpenAICompatibleBackend",
    "ClaudeCLIBackend",
    "CompletionResult",
    "StructuredResponse",
    "STRUCTURED_RESPONSE_SCHEMA",
    "TokenUsage",
    "CompositeContext",
    "LazyContext",
    "StringContext",
    "REPLEnv",
    "REPLResult",
    "RLMConfig",
    "DefaultsConfig",
    "RoleConfig",
    "ResolvedRoleConfig",
    "SettingsConfig",
    "ConfigError",
    "load_config",
]
