"""Unit tests for rlm.repl module."""

from __future__ import annotations

from pathlib import Path

from rlm.context import LazyContext, StringContext
from rlm.repl import REPLEnv, REPLResult

from .conftest import SAMPLE_TEXT, noop_llm_query

# ---------------------------------------------------------------------------
# REPLResult dataclass
# ---------------------------------------------------------------------------


class TestREPLResult:
    """Tests for the REPLResult dataclass."""

    def test_default_values(self) -> None:
        result = REPLResult(output="")
        assert result.output == ""
        assert result.final_answer is None
        assert result.error is None
        assert result.success is True

    def test_custom_initialization(self) -> None:
        result = REPLResult(
            output="hello",
            final_answer="answer",
            error="oops",
            success=False,
        )
        assert result.output == "hello"
        assert result.final_answer == "answer"
        assert result.error == "oops"
        assert result.success is False


# ---------------------------------------------------------------------------
# REPLEnv
# ---------------------------------------------------------------------------


class TestREPLEnvBasicExecution:
    """Tests for basic code execution in REPLEnv."""

    def test_simple_print(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("hello")')
        assert result.success is True
        assert result.output.strip() == "hello"

    def test_arithmetic(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("x = 2 + 3\nprint(x)")
        assert result.success is True
        assert "5" in result.output


class TestREPLEnvContextAccess:
    """Tests for CONTEXT variable access."""

    def test_context_length(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(len(CONTEXT))")
        assert result.success is True
        assert str(len(SAMPLE_TEXT)) in result.output

    def test_context_slicing(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(CONTEXT[:9])")
        assert result.success is True
        assert "Chapter 1" in result.output

    def test_context_findall(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('matches = CONTEXT.findall(r"Chapter \\d+")\nprint(matches)')
        assert result.success is True
        assert "Chapter 1" in result.output
        assert "Chapter 4" in result.output

    def test_context_search(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute(
            'match = CONTEXT.search(r"Chapter (\\d+)")\nprint(match.group(1))'
        )
        assert result.success is True
        assert "1" in result.output

    def test_context_lines(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute(
            "lines = list(CONTEXT.lines())\nprint(len(lines))\nprint(lines[0])"
        )
        assert result.success is True
        assert "Chapter 1: Introduction" in result.output


class TestREPLEnvPrintCapture:
    """Tests for print output capture."""

    def test_single_print(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("one")')
        assert result.output == "one\n"

    def test_multiple_prints(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("a")\nprint("b")\nprint("c")')
        assert result.output == "a\nb\nc\n"

    def test_print_with_sep(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("x", "y", "z", sep="-")')
        assert result.output == "x-y-z\n"

    def test_print_with_end(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("hello", end="!")')
        assert result.output == "hello!"


class TestREPLEnvFinal:
    """Tests for FINAL() and FINAL_VAR()."""

    def test_final_sets_answer(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('FINAL("the answer")')
        assert result.success is True
        assert result.final_answer == "the answer"

    def test_final_var_sets_answer(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('my_result = "computed"\nFINAL_VAR("my_result")')
        assert result.success is True
        assert result.final_answer == "computed"

    def test_final_var_missing_variable(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('FINAL_VAR("nonexistent")')
        assert result.success is False
        assert "nonexistent" in (result.error or "")
        assert "not found" in (result.error or "")

    def test_fallback_final_prefix_variable(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('final_summary = "auto-detected answer"')
        assert result.success is True
        assert result.final_answer == "auto-detected answer"


class TestREPLEnvNamespacePersistence:
    """Tests for namespace persistence across execute() calls."""

    def test_variables_survive_across_calls(self, repl_env: REPLEnv) -> None:
        repl_env.execute("x = 42")
        result = repl_env.execute("print(x)")
        assert result.success is True
        assert "42" in result.output


class TestREPLEnvLlmQuery:
    """Tests for llm_query integration."""

    def test_llm_query_calls_provided_function(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('response = llm_query("test prompt")\nprint(response)')
        assert result.success is True
        assert "[mock response to:" in result.output


class TestREPLEnvSafeBuiltins:
    """Tests for safe builtin restrictions."""

    def test_import_not_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("import os")
        assert result.success is False

    def test_open_not_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('open("/etc/passwd")')
        assert result.success is False

    def test_exec_not_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('exec("x = 1")')
        assert result.success is False

    def test_eval_not_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('eval("1 + 1")')
        assert result.success is False

    def test_compile_not_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('compile("pass", "<string>", "exec")')
        assert result.success is False

    def test_len_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print(len("abc"))')
        assert result.success is True
        assert "3" in result.output

    def test_sorted_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(sorted([3, 1, 2]))")
        assert result.success is True
        assert "[1, 2, 3]" in result.output

    def test_enumerate_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(list(enumerate(['a', 'b'])))")
        assert result.success is True
        assert "(0, 'a')" in result.output

    def test_isinstance_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(isinstance(42, int))")
        assert result.success is True
        assert "True" in result.output


class TestREPLEnvOutputTruncation:
    """Tests for max_output_length truncation."""

    def test_output_truncated_when_exceeding_limit(self) -> None:
        env = REPLEnv(
            context="test",
            llm_query_fn=noop_llm_query,
            max_output_length=50,
        )
        result = env.execute('print("x" * 100)')
        assert result.success is True
        assert "[...output truncated]" in result.output
        # The output should be max_output_length + truncation message
        assert len(result.output) < 200


class TestREPLEnvErrorHandling:
    """Tests for error handling in execute()."""

    def test_syntax_error(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("def foo(")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("1 / 0")
        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(undefined_var)")
        assert result.success is False
        assert result.error is not None
        assert "NameError" in result.error


class TestREPLEnvContextTypeCoercion:
    """Tests for context type coercion in __init__."""

    def test_str_wraps_in_string_context(self) -> None:
        env = REPLEnv(context="hello", llm_query_fn=noop_llm_query)
        assert isinstance(env.context, StringContext)

    def test_path_wraps_in_lazy_context(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_text("content", encoding="utf-8")
        env = REPLEnv(context=p, llm_query_fn=noop_llm_query)
        assert isinstance(env.context, LazyContext)
        env.context.close()


class TestREPLEnvPreImportedModules:
    """Tests for pre-imported modules in the REPL namespace."""

    def test_re_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print(re.findall(r"\\d+", "abc123def456"))')
        assert result.success is True
        assert "123" in result.output

    def test_json_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print(json.dumps({"a": 1}))')
        assert result.success is True
        assert '"a"' in result.output

    def test_math_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(math.sqrt(16))")
        assert result.success is True
        assert "4.0" in result.output

    def test_collections_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('c = collections.Counter("aabbc")\nprint(c.most_common(1))')
        assert result.success is True

    def test_itertools_available(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(list(itertools.chain([1], [2], [3])))")
        assert result.success is True
        assert "[1, 2, 3]" in result.output
