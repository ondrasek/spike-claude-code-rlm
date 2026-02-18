"""Unit tests for rlm.context module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rlm.context import CompositeContext, LazyContext, StringContext

# conftest provides: SAMPLE_TEXT, SAMPLE_TEXT_SMALL, SAMPLE_TEXT_MULTILINE,
# tmp_text_file, tmp_empty_file, tmp_multifile_dir,
# string_context, lazy_context, empty_lazy_context, composite_context
from .conftest import SAMPLE_TEXT

# =========================================================================
# LazyContext
# =========================================================================


class TestLazyContextInit:
    def test_init_opens_file(self, tmp_text_file: Path) -> None:
        ctx = LazyContext(tmp_text_file)
        assert len(ctx) > 0
        ctx.close()

    def test_init_empty_file(self, tmp_empty_file: Path) -> None:
        ctx = LazyContext(tmp_empty_file)
        assert len(ctx) == 0
        ctx.close()


class TestLazyContextLen:
    def test_len_returns_byte_size(self, lazy_context: LazyContext) -> None:
        expected = len(SAMPLE_TEXT.encode("utf-8"))
        assert len(lazy_context) == expected

    def test_len_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert len(empty_lazy_context) == 0


class TestLazyContextGetitem:
    def test_getitem_int_index(self, lazy_context: LazyContext) -> None:
        assert lazy_context[0] == SAMPLE_TEXT[0]

    def test_getitem_slice(self, lazy_context: LazyContext) -> None:
        assert lazy_context[0:10] == SAMPLE_TEXT[0:10]

    def test_getitem_slice_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert empty_lazy_context[0:10] == ""

    def test_getitem_int_empty_file_raises(self, empty_lazy_context: LazyContext) -> None:
        with pytest.raises(IndexError, match="empty context"):
            empty_lazy_context[0]


class TestLazyContextStr:
    def test_str_returns_full_content(self, lazy_context: LazyContext) -> None:
        assert str(lazy_context) == SAMPLE_TEXT

    def test_str_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert str(empty_lazy_context) == ""


class TestLazyContextContains:
    def test_contains_present_substring(self, lazy_context: LazyContext) -> None:
        assert "Chapter 1" in lazy_context

    def test_contains_absent_substring(self, lazy_context: LazyContext) -> None:
        assert "nonexistent" not in lazy_context

    def test_contains_non_string(self, lazy_context: LazyContext) -> None:
        assert 42 not in lazy_context  # type: ignore[operator]

    def test_contains_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert "anything" not in empty_lazy_context


class TestLazyContextRepr:
    def test_repr_format(self, lazy_context: LazyContext) -> None:
        r = repr(lazy_context)
        assert r.startswith("LazyContext(")
        assert "size=" in r


class TestLazyContextSearch:
    def test_search_returns_match(self, lazy_context: LazyContext) -> None:
        m = lazy_context.search(r"Chapter \d+")
        assert m is not None
        assert isinstance(m, re.Match)
        # LazyContext search returns Match[bytes]
        assert isinstance(m.group(), bytes)

    def test_search_no_match(self, lazy_context: LazyContext) -> None:
        assert lazy_context.search(r"ZZZZZ") is None

    def test_search_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert empty_lazy_context.search(r"anything") is None

    def test_search_with_flags(self, lazy_context: LazyContext) -> None:
        m = lazy_context.search(r"chapter", re.IGNORECASE)
        assert m is not None


class TestLazyContextFindall:
    def test_findall_returns_decoded_strings(self, lazy_context: LazyContext) -> None:
        matches = lazy_context.findall(r"Chapter \d+")
        assert isinstance(matches, list)
        assert len(matches) == 4
        assert all(isinstance(m, str) for m in matches)
        assert matches[0] == "Chapter 1"

    def test_findall_no_match(self, lazy_context: LazyContext) -> None:
        assert lazy_context.findall(r"ZZZZZ") == []

    def test_findall_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert empty_lazy_context.findall(r"anything") == []


class TestLazyContextLines:
    def test_lines_yields_stripped_lines(self, lazy_context: LazyContext) -> None:
        result = list(lazy_context.lines())
        assert len(result) > 0
        # Lines should not have trailing newlines
        for line in result:
            assert not line.endswith("\n")
        assert result[0] == "Chapter 1: Introduction"

    def test_lines_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert list(empty_lazy_context.lines()) == []


class TestLazyContextChunk:
    def test_chunk_returns_decoded_substring(self, lazy_context: LazyContext) -> None:
        chunk = lazy_context.chunk(0, 9)
        assert chunk == SAMPLE_TEXT[0:9]

    def test_chunk_empty_file(self, empty_lazy_context: LazyContext) -> None:
        assert empty_lazy_context.chunk(0, 10) == ""


class TestLazyContextClose:
    def test_close_releases_resources(self, tmp_text_file: Path) -> None:
        ctx = LazyContext(tmp_text_file)
        ctx.close()
        # After close, _mm should be None and _fd should be -1
        assert ctx._mm is None
        assert ctx._fd == -1

    def test_double_close_is_safe(self, tmp_text_file: Path) -> None:
        ctx = LazyContext(tmp_text_file)
        ctx.close()
        ctx.close()  # Should not raise


class TestLazyContextManager:
    def test_context_manager(self, tmp_text_file: Path) -> None:
        with LazyContext(tmp_text_file) as ctx:
            assert len(ctx) > 0
        assert ctx._mm is None


# =========================================================================
# StringContext
# =========================================================================


class TestStringContextLen:
    def test_len_returns_character_count(self, string_context: StringContext) -> None:
        assert len(string_context) == len(SAMPLE_TEXT)


class TestStringContextGetitem:
    def test_getitem_int(self, string_context: StringContext) -> None:
        assert string_context[0] == SAMPLE_TEXT[0]

    def test_getitem_slice(self, string_context: StringContext) -> None:
        assert string_context[0:10] == SAMPLE_TEXT[0:10]


class TestStringContextStr:
    def test_str_returns_text(self, string_context: StringContext) -> None:
        assert str(string_context) == SAMPLE_TEXT


class TestStringContextContains:
    def test_contains_present(self, string_context: StringContext) -> None:
        assert "Chapter 1" in string_context

    def test_contains_absent(self, string_context: StringContext) -> None:
        assert "nonexistent" not in string_context

    def test_contains_non_string(self, string_context: StringContext) -> None:
        assert 42 not in string_context  # type: ignore[operator]


class TestStringContextRepr:
    def test_repr_format(self, string_context: StringContext) -> None:
        r = repr(string_context)
        assert r.startswith("StringContext(")
        assert "size=" in r


class TestStringContextSearch:
    def test_search_returns_str_match(self, string_context: StringContext) -> None:
        m = string_context.search(r"Chapter \d+")
        assert m is not None
        assert isinstance(m, re.Match)
        # StringContext search returns Match[str], not Match[bytes]
        assert isinstance(m.group(), str)
        assert m.group() == "Chapter 1"

    def test_search_no_match(self, string_context: StringContext) -> None:
        assert string_context.search(r"ZZZZZ") is None

    def test_search_with_flags(self, string_context: StringContext) -> None:
        m = string_context.search(r"chapter", re.IGNORECASE)
        assert m is not None


class TestStringContextFindall:
    def test_findall_returns_strings(self, string_context: StringContext) -> None:
        matches = string_context.findall(r"Chapter \d+")
        assert len(matches) == 4
        assert all(isinstance(m, str) for m in matches)

    def test_findall_no_match(self, string_context: StringContext) -> None:
        assert string_context.findall(r"ZZZZZ") == []


class TestStringContextLines:
    def test_lines_yields_lines(self, string_context: StringContext) -> None:
        result = list(string_context.lines())
        assert len(result) > 0
        assert result[0] == "Chapter 1: Introduction"

    def test_lines_multiline(self) -> None:
        ctx = StringContext("line1\nline2\nline3")
        assert list(ctx.lines()) == ["line1", "line2", "line3"]


class TestStringContextChunk:
    def test_chunk_returns_substring(self, string_context: StringContext) -> None:
        chunk = string_context.chunk(0, 9)
        assert chunk == SAMPLE_TEXT[0:9]


class TestStringContextClose:
    def test_close_is_noop(self) -> None:
        ctx = StringContext("hello")
        ctx.close()  # Should not raise
        # Text should still be accessible after close (no-op)
        assert str(ctx) == "hello"


# =========================================================================
# CompositeContext
# =========================================================================


class TestCompositeContextInit:
    def test_empty_sources_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            CompositeContext({})


class TestCompositeContextFromPaths:
    def test_from_paths_creates_context(self, tmp_multifile_dir: Path) -> None:
        paths = sorted(tmp_multifile_dir.glob("*.txt"))
        ctx = CompositeContext.from_paths(paths)
        assert len(ctx.files) == len(paths)
        ctx.close()

    def test_from_paths_duplicate_basename_disambiguation(self, tmp_path: Path) -> None:
        sub1 = tmp_path / "dir1"
        sub2 = tmp_path / "dir2"
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / "data.txt").write_text("aaa", encoding="utf-8")
        (sub2 / "data.txt").write_text("bbb", encoding="utf-8")
        ctx = CompositeContext.from_paths([sub1 / "data.txt", sub2 / "data.txt"])
        assert len(ctx.files) == 2
        # One key should be "data.txt", the other the full path
        assert "data.txt" in ctx.files
        ctx.close()


class TestCompositeContextFromDirectory:
    def test_from_directory_finds_all_files(self, tmp_multifile_dir: Path) -> None:
        ctx = CompositeContext.from_directory(tmp_multifile_dir)
        # Should find file_a.txt, file_b.txt, sub/file_c.txt
        assert len(ctx.files) == 3
        ctx.close()

    def test_from_directory_with_glob(self, tmp_multifile_dir: Path) -> None:
        ctx = CompositeContext.from_directory(tmp_multifile_dir, glob="*.txt")
        # Only top-level *.txt, not sub/file_c.txt
        assert len(ctx.files) == 2
        ctx.close()

    def test_from_directory_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No files matched"):
            CompositeContext.from_directory(tmp_path, glob="*.xyz")


class TestCompositeContextFiles:
    def test_files_property(self, composite_context: CompositeContext) -> None:
        files = composite_context.files
        assert isinstance(files, list)
        assert len(files) == 3

    def test_file_access(self, composite_context: CompositeContext) -> None:
        name = composite_context.files[0]
        src = composite_context.file(name)
        assert len(src) > 0

    def test_file_missing_raises_key_error(self, composite_context: CompositeContext) -> None:
        with pytest.raises(KeyError):
            composite_context.file("no_such_file.txt")


class TestCompositeContextLen:
    def test_len_sums_all_sources(self, composite_context: CompositeContext) -> None:
        total = sum(len(composite_context.file(f)) for f in composite_context.files)
        assert len(composite_context) == total


class TestCompositeContextContains:
    def test_contains_across_files(self, composite_context: CompositeContext) -> None:
        assert "file A" in composite_context
        assert "file B" in composite_context
        assert "file C" in composite_context

    def test_contains_absent(self, composite_context: CompositeContext) -> None:
        assert "nonexistent_xyz" not in composite_context

    def test_contains_non_string(self, composite_context: CompositeContext) -> None:
        assert 42 not in composite_context  # type: ignore[operator]


class TestCompositeContextStr:
    def test_str_concatenates_with_separators(self, composite_context: CompositeContext) -> None:
        result = str(composite_context)
        # Should contain separator markers
        assert "---" in result
        # Should contain content from all files
        assert "file A" in result
        assert "file B" in result
        assert "file C" in result


class TestCompositeContextGetitem:
    def test_getitem_slice(self, composite_context: CompositeContext) -> None:
        full = str(composite_context)
        assert composite_context[:20] == full[:20]

    def test_getitem_int(self, composite_context: CompositeContext) -> None:
        full = str(composite_context)
        assert composite_context[0] == full[0]


class TestCompositeContextSearch:
    def test_search_returns_filename_and_match(self, composite_context: CompositeContext) -> None:
        result = composite_context.search(r"file [ABC]")
        assert result is not None
        filename, match = result
        assert isinstance(filename, str)
        assert isinstance(match, re.Match)

    def test_search_no_match(self, composite_context: CompositeContext) -> None:
        assert composite_context.search(r"ZZZZZ") is None


class TestCompositeContextFindall:
    def test_findall_returns_filename_match_tuples(
        self, composite_context: CompositeContext
    ) -> None:
        results = composite_context.findall(r"Contents of file [A-C]")
        assert len(results) == 3
        # Each result is (filename, match_text)
        for filename, match_text in results:
            assert isinstance(filename, str)
            assert isinstance(match_text, str)
            assert match_text.startswith("Contents of file")


class TestCompositeContextLines:
    def test_lines_yields_filename_line_tuples(self, composite_context: CompositeContext) -> None:
        result = list(composite_context.lines())
        assert len(result) > 0
        for filename, line in result:
            assert isinstance(filename, str)
            assert isinstance(line, str)


class TestCompositeContextChunk:
    def test_chunk_on_concatenated_view(self, composite_context: CompositeContext) -> None:
        full = str(composite_context)
        assert composite_context.chunk(0, 15) == full[0:15]


class TestCompositeContextClose:
    def test_close_closes_all_sources(self, tmp_multifile_dir: Path) -> None:
        ctx = CompositeContext.from_directory(tmp_multifile_dir)
        ctx.close()
        # All underlying LazyContexts should be closed
        for src in ctx._sources.values():
            if isinstance(src, LazyContext):
                assert src._mm is None


class TestCompositeContextManager:
    def test_context_manager(self, tmp_multifile_dir: Path) -> None:
        with CompositeContext.from_directory(tmp_multifile_dir) as ctx:
            assert len(ctx.files) == 3
        # After exit, sources should be closed
        for src in ctx._sources.values():
            if isinstance(src, LazyContext):
                assert src._mm is None
