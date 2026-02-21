"""LLM smoke tests — verify real models generate working REPL code.

These tests run against a live Ollama server and validate that the model:
- Calls the correct CONTEXT API methods
- Uses llm_query() for text analysis when appropriate
- Produces a final answer via FINAL() within bounded iterations
- Returns answers containing expected keywords

Skipped automatically when Ollama is unreachable.

Run explicitly::

    OLLAMA_HOST=host.docker.internal:11434 pytest -m ollama -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rlm.backends import OpenAICompatibleBackend
from rlm.rlm import RLM, RLMResult

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
_OLLAMA_BASE_URL = (
    f"{_OLLAMA_HOST}/v1" if _OLLAMA_HOST.startswith("http") else f"http://{_OLLAMA_HOST}/v1"
)
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")

CONSTITUTION_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "01-constitution-analysis"
    / "constitution.txt"
)


def _ollama_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{_OLLAMA_BASE_URL.rstrip('/v1')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not reachable",
)

skip_no_constitution = pytest.mark.skipif(
    not CONSTITUTION_PATH.exists(),
    reason="Constitution text not found",
)


def _make_rlm() -> RLM:
    """Create an RLM instance configured for Ollama smoke tests."""
    backend = OpenAICompatibleBackend(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    return RLM(
        backend=backend,
        model=_OLLAMA_MODEL,
        max_iterations=5,
        max_tokens=4096,
        verbose=True,
    )


def _run_query(query: str) -> RLMResult:
    """Run a query against the constitution and return the result."""
    rlm = _make_rlm()
    return rlm.completion(context=CONSTITUTION_PATH, query=query)


# ---------------------------------------------------------------------------
# Query 1: Document structure (findall only, no recursive calls)
# ---------------------------------------------------------------------------

_QUERY_STRUCTURE = (
    "Find the major sections of this document. "
    "Note: Article headings use mixed case (e.g. 'Article 1', 'ARTICLE TWO'). "
    r"Step 1: Use CONTEXT.findall(r'^ARTICLE\s+\S+$', re.MULTILINE | re.IGNORECASE) "
    "to find Article headings. "
    r"Step 2: Use CONTEXT.findall(r'^Amendment [IVXLC]+', re.MULTILINE) "
    "to find Amendment headings. "
    "Step 3: Print both lists. "
    "Step 4: Call FINAL() with a numbered list of every heading you found, "
    "Articles first then Amendments."
)


@pytest.fixture(scope="module")
def structure_result() -> RLMResult:
    """Run the document structure query once, share across all tests."""
    return _run_query(_QUERY_STRUCTURE)


@pytest.mark.slow
@pytest.mark.ollama
@skip_no_ollama
@skip_no_constitution
class TestDocumentStructure:
    """Query 1: Find Articles and Amendments via CONTEXT.findall()."""

    def test_completes_successfully(self, structure_result: RLMResult) -> None:
        assert structure_result.success, f"RLM failed: {structure_result.error}"

    def test_finds_articles(self, structure_result: RLMResult) -> None:
        assert structure_result.success
        assert "ARTICLE" in structure_result.answer.upper()

    def test_finds_amendments(self, structure_result: RLMResult) -> None:
        assert structure_result.success
        assert "AMENDMENT" in structure_result.answer.upper()

    def test_bounded_iterations(self, structure_result: RLMResult) -> None:
        assert structure_result.success
        assert structure_result.stats.iterations <= 3


# ---------------------------------------------------------------------------
# Query 2: Bill of Rights (chunk + llm_query)
# ---------------------------------------------------------------------------

_QUERY_BILL_OF_RIGHTS = (
    "What rights does the Bill of Rights protect? "
    "The Bill of Rights text starts around byte offset 27000 and runs about 2700 bytes. "
    "Step 1: Grab the chunk with chunk = CONTEXT.chunk(27000, 2800) and print it to verify. "
    "Step 2: Pass the chunk to llm_query("
    "'Summarize each of the 10 amendments (I-X) in one sentence each.'). "
    "Step 3: Call FINAL() with the result from llm_query."
)


@pytest.fixture(scope="module")
def bill_of_rights_result() -> RLMResult:
    """Run the Bill of Rights query once, share across all tests."""
    return _run_query(_QUERY_BILL_OF_RIGHTS)


@pytest.mark.slow
@pytest.mark.ollama
@skip_no_ollama
@skip_no_constitution
class TestBillOfRights:
    """Query 2: Extract and summarize the Bill of Rights via chunk + llm_query."""

    def test_completes_successfully(self, bill_of_rights_result: RLMResult) -> None:
        assert bill_of_rights_result.success, f"RLM failed: {bill_of_rights_result.error}"

    def test_uses_recursive_call(self, bill_of_rights_result: RLMResult) -> None:
        assert bill_of_rights_result.success
        assert bill_of_rights_result.stats.recursive_calls >= 1, (
            "Expected at least one llm_query() call"
        )

    def test_mentions_speech_or_religion(self, bill_of_rights_result: RLMResult) -> None:
        """Amendment I protects speech and religion — answer should mention at least one."""
        assert bill_of_rights_result.success
        answer_lower = bill_of_rights_result.answer.lower()
        assert "speech" in answer_lower or "religion" in answer_lower

    def test_mentions_arms(self, bill_of_rights_result: RLMResult) -> None:
        """Amendment II protects the right to bear arms."""
        assert bill_of_rights_result.success
        assert "arms" in bill_of_rights_result.answer.lower()

    def test_bounded_iterations(self, bill_of_rights_result: RLMResult) -> None:
        assert bill_of_rights_result.success
        assert bill_of_rights_result.stats.iterations <= 3


# ---------------------------------------------------------------------------
# Query 3: Congressional powers (chunk + llm_query)
# ---------------------------------------------------------------------------

_QUERY_CONGRESSIONAL_POWERS = (
    "What specific powers does Congress have? "
    "Section 8 of Article 1 starts around byte offset 8500 and runs about 2600 bytes. "
    "Step 1: Grab the chunk with chunk = CONTEXT.chunk(8500, 2700) and print it to verify. "
    "Step 2: Pass the chunk to llm_query("
    "'List each enumerated power of Congress in this text.'). "
    "Step 3: Call FINAL() with the result from llm_query."
)


@pytest.fixture(scope="module")
def congressional_powers_result() -> RLMResult:
    """Run the Congressional powers query once, share across all tests."""
    return _run_query(_QUERY_CONGRESSIONAL_POWERS)


@pytest.mark.slow
@pytest.mark.ollama
@skip_no_ollama
@skip_no_constitution
class TestCongressionalPowers:
    """Query 3: Extract Section 8 and list enumerated powers via llm_query."""

    def test_completes_successfully(self, congressional_powers_result: RLMResult) -> None:
        assert congressional_powers_result.success, (
            f"RLM failed: {congressional_powers_result.error}"
        )

    def test_uses_recursive_call(self, congressional_powers_result: RLMResult) -> None:
        assert congressional_powers_result.success
        assert congressional_powers_result.stats.recursive_calls >= 1, (
            "Expected at least one llm_query() call"
        )

    def test_mentions_tax(self, congressional_powers_result: RLMResult) -> None:
        """Power #1 is to lay and collect Taxes."""
        assert congressional_powers_result.success
        assert "tax" in congressional_powers_result.answer.lower()

    def test_mentions_commerce(self, congressional_powers_result: RLMResult) -> None:
        """Power #3 is to regulate Commerce."""
        assert congressional_powers_result.success
        assert "commerce" in congressional_powers_result.answer.lower()

    def test_mentions_war(self, congressional_powers_result: RLMResult) -> None:
        """Power #11 is to declare War."""
        assert congressional_powers_result.success
        assert "war" in congressional_powers_result.answer.lower()

    def test_bounded_iterations(self, congressional_powers_result: RLMResult) -> None:
        assert congressional_powers_result.success
        assert congressional_powers_result.stats.iterations <= 3
