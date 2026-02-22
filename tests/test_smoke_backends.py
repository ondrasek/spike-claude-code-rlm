"""Live smoke tests for OpenAI, OpenRouter, and Hugging Face backends.

Each test class targets a single backend and is gated by the corresponding
API key environment variable.  Keys are loaded from ``.env`` via
``python-dotenv`` (see ``conftest.py``).  Tests are marked ``slow`` plus a
per-backend marker so they can be run selectively::

    pytest -m openai -v
    pytest -m openrouter -v
    pytest -m huggingface -v

Skipped automatically when the required environment variable is absent.
"""

from __future__ import annotations

import os

import pytest

from rlm.backends import OpenAICompatibleBackend
from rlm.cli import _BACKEND_DEFAULT_MODELS
from rlm.rlm import RLM, RLMResult

# ---------------------------------------------------------------------------
# Shared sample context (small — keeps cost/latency low)
# ---------------------------------------------------------------------------

_SAMPLE_CONTEXT = """\
Chapter 1: Introduction
This document describes the history of space exploration.
Key milestones include the launch of Sputnik, the Apollo Moon landings,
the Space Shuttle programme, and the International Space Station.

Chapter 2: Early Exploration
Sputnik was launched in 1957 by the Soviet Union.
Yuri Gagarin became the first human in space in 1961.

Chapter 3: The Moon Race
The Apollo 11 mission landed on the Moon on July 20, 1969.
Neil Armstrong and Buzz Aldrin walked on the lunar surface.

Chapter 4: Modern Era
The International Space Station has been continuously occupied since 2000.
SpaceX and other commercial companies now launch payloads regularly.
"""

_QUERY = (
    "List the major milestones mentioned in this document. "
    "Use CONTEXT[:500] to read the text, then call FINAL() with a short bullet list."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_env(env_var: str) -> str:
    """Return the env var value or skip the test at runtime.

    Using a runtime check (instead of ``pytest.mark.skipif``) ensures
    that ``.env`` has already been loaded by ``pytest_configure`` in
    ``conftest.py`` before the check runs.
    """
    value = os.getenv(env_var)
    if not value:
        pytest.skip(f"{env_var} not set")
    return value


def _run_smoke(backend_name: str, env_var: str, base_url: str) -> RLMResult:
    """Run a single RLM completion against a live backend."""
    api_key = _require_env(env_var)
    model = _BACKEND_DEFAULT_MODELS[backend_name]
    backend = OpenAICompatibleBackend(base_url=base_url, api_key=api_key)
    rlm = RLM(backend=backend, model=model, max_iterations=5, max_tokens=2048, verbose=True)
    return rlm.completion(context=_SAMPLE_CONTEXT, query=_QUERY)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_result() -> RLMResult:
    """Run the smoke query once via OpenAI, share across tests."""
    return _run_smoke("openai", "OPENAI_API_KEY", "https://api.openai.com/v1")


@pytest.mark.slow
@pytest.mark.openai
class TestOpenAISmoke:
    """Smoke tests against the live OpenAI API."""

    def test_completes_successfully(self, openai_result: RLMResult) -> None:
        assert openai_result.success, f"RLM failed: {openai_result.error}"

    def test_answer_mentions_moon(self, openai_result: RLMResult) -> None:
        assert openai_result.success
        assert "moon" in openai_result.answer.lower()

    def test_bounded_iterations(self, openai_result: RLMResult) -> None:
        assert openai_result.success
        assert openai_result.stats.iterations <= 4


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openrouter_result() -> RLMResult:
    """Run the smoke query once via OpenRouter, share across tests."""
    return _run_smoke("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")


@pytest.mark.slow
@pytest.mark.openrouter
class TestOpenRouterSmoke:
    """Smoke tests against the live OpenRouter API."""

    def test_completes_successfully(self, openrouter_result: RLMResult) -> None:
        assert openrouter_result.success, f"RLM failed: {openrouter_result.error}"

    def test_answer_mentions_moon(self, openrouter_result: RLMResult) -> None:
        assert openrouter_result.success
        assert "moon" in openrouter_result.answer.lower()

    def test_bounded_iterations(self, openrouter_result: RLMResult) -> None:
        assert openrouter_result.success
        assert openrouter_result.stats.iterations <= 4


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def huggingface_result() -> RLMResult:
    """Run the smoke query once via Hugging Face, share across tests."""
    return _run_smoke("huggingface", "HF_TOKEN", "https://router.huggingface.co/v1")


@pytest.mark.slow
@pytest.mark.huggingface
class TestHuggingFaceSmoke:
    """Smoke tests against the live Hugging Face Inference API."""

    def test_completes_successfully(self, huggingface_result: RLMResult) -> None:
        assert huggingface_result.success, f"RLM failed: {huggingface_result.error}"

    def test_answer_mentions_moon(self, huggingface_result: RLMResult) -> None:
        assert huggingface_result.success
        assert "moon" in huggingface_result.answer.lower()

    def test_bounded_iterations(self, huggingface_result: RLMResult) -> None:
        assert huggingface_result.success
        assert huggingface_result.stats.iterations <= 4
