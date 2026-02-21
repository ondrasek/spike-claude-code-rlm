"""Unit tests for rlm.repl module."""

from __future__ import annotations

from pathlib import Path

from rlm.context import CompositeContext
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
    """Tests for CONTEXT variable access — CONTEXT is a plain Python str."""

    def test_context_is_plain_str(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(type(CONTEXT).__name__)")
        assert result.success is True
        assert "str" in result.output

    def test_context_length(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(len(CONTEXT))")
        assert result.success is True
        assert str(len(SAMPLE_TEXT)) in result.output

    def test_context_slicing(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("print(CONTEXT[:9])")
        assert result.success is True
        assert "Chapter 1" in result.output

    def test_context_re_findall(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('matches = re.findall(r"Chapter \\d+", CONTEXT)\nprint(matches)')
        assert result.success is True
        assert "Chapter 1" in result.output
        assert "Chapter 4" in result.output

    def test_context_re_search(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute(
            'match = re.search(r"Chapter (\\d+)", CONTEXT)\nprint(match.group(1))'
        )
        assert result.success is True
        assert "1" in result.output

    def test_context_splitlines(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute(
            "lines = CONTEXT.splitlines()\nprint(len(lines))\nprint(lines[0])"
        )
        assert result.success is True
        assert "Chapter 1: Introduction" in result.output

    def test_context_split_newline(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute("lines = CONTEXT.split('\\n')\nprint(lines[0])")
        assert result.success is True
        assert "Chapter 1: Introduction" in result.output

    def test_context_contains(self, repl_env: REPLEnv) -> None:
        result = repl_env.execute('print("Chapter 1" in CONTEXT)')
        assert result.success is True
        assert "True" in result.output


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

    def test_str_produces_plain_str_context(self) -> None:
        env = REPLEnv(context="hello", llm_query_fn=noop_llm_query)
        assert isinstance(env.context_str, str)
        assert env.context_str == "hello"
        assert env.files_dict is None

    def test_path_produces_plain_str_context(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_text("content", encoding="utf-8")
        env = REPLEnv(context=p, llm_query_fn=noop_llm_query)
        assert isinstance(env.context_str, str)
        assert env.context_str == "content"
        assert env.files_dict is None

    def test_context_str_attribute(self) -> None:
        env = REPLEnv(context="hello world", llm_query_fn=noop_llm_query)
        assert type(env.context_str) is str
        assert env.context_str == "hello world"


class TestREPLEnvShowVars:
    """Tests for SHOW_VARS() helper."""

    def test_show_vars_empty(self) -> None:
        env = REPLEnv(context="test", llm_query_fn=noop_llm_query)
        result = env.execute("SHOW_VARS()")
        assert result.success is True
        assert "no user-defined variables" in result.output

    def test_show_vars_with_user_variables(self) -> None:
        env = REPLEnv(context="test", llm_query_fn=noop_llm_query)
        env.execute("x = 42\ny = 'hello'")
        result = env.execute("SHOW_VARS()")
        assert result.success is True
        assert "x = 42" in result.output
        assert "y = 'hello'" in result.output

    def test_show_vars_excludes_internals(self) -> None:
        env = REPLEnv(context="test", llm_query_fn=noop_llm_query)
        env.execute("my_var = 'visible'")
        result = env.execute("SHOW_VARS()")
        assert result.success is True
        assert "my_var" in result.output
        # REPL internals should NOT appear
        assert "CONTEXT" not in result.output
        assert "FINAL" not in result.output
        assert "llm_query" not in result.output


class TestREPLEnvFilesDict:
    """Tests for FILES dict with multi-file context."""

    def test_files_dict_present_for_composite(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("content B", encoding="utf-8")
        ctx = CompositeContext.from_paths([tmp_path / "a.txt", tmp_path / "b.txt"])
        env = REPLEnv(context=ctx, llm_query_fn=noop_llm_query)
        assert env.files_dict is not None
        assert "a.txt" in env.files_dict
        assert "b.txt" in env.files_dict
        assert env.files_dict["a.txt"] == "content A"
        assert env.files_dict["b.txt"] == "content B"
        ctx.close()

    def test_files_available_in_namespace(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("hello X", encoding="utf-8")
        (tmp_path / "y.txt").write_text("hello Y", encoding="utf-8")
        ctx = CompositeContext.from_paths([tmp_path / "x.txt", tmp_path / "y.txt"])
        env = REPLEnv(context=ctx, llm_query_fn=noop_llm_query)
        result = env.execute("print(list(FILES.keys()))")
        assert result.success is True
        assert "x.txt" in result.output
        assert "y.txt" in result.output
        ctx.close()

    def test_files_not_present_for_single_file(self) -> None:
        env = REPLEnv(context="single file content", llm_query_fn=noop_llm_query)
        assert env.files_dict is None
        result = env.execute('print("FILES" in dir())')
        assert result.success is True
        assert "False" in result.output

    def test_files_from_path_list(self, tmp_path: Path) -> None:
        (tmp_path / "p.txt").write_text("path content", encoding="utf-8")
        env = REPLEnv(
            context=[tmp_path / "p.txt"],
            llm_query_fn=noop_llm_query,
        )
        assert env.files_dict is not None
        assert "p.txt" in env.files_dict


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
