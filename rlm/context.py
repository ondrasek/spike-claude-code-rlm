"""Lazy, memory-mapped context for RLM.

Provides context wrappers so REPL code can use ``CONTEXT.findall()`` /
``CONTEXT.search()`` regardless of the underlying storage:

* :class:`LazyContext` — memory-maps a single file via :mod:`mmap`.
* :class:`StringContext` — wraps an in-memory ``str``.
* :class:`CompositeContext` — wraps *multiple* files and/or strings,
  providing per-file access (``CONTEXT.file("name")``) and cross-file
  search (``CONTEXT.findall(pattern)``).
"""

from __future__ import annotations

import mmap
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import overload


class LazyContext:
    """Memory-mapped, read-only view of a text file.

    The file is mapped into virtual memory via :mod:`mmap`.  The OS will
    page regions in and out transparently, so physical RAM usage stays
    bounded even for multi-gigabyte files.

    The class exposes a string-like interface sufficient for the patterns
    used in the RLM REPL:

    * Slicing (``CONTEXT[100:200]``) returns ``str``
    * ``len(CONTEXT)`` returns the *byte* length (≈ character length for
      ASCII / UTF-8 without multi-byte sequences)
    * ``CONTEXT.search(pattern)`` / ``CONTEXT.findall(pattern)`` wrap
      :func:`re.search` / :func:`re.findall` operating on the raw bytes
    * ``CONTEXT.lines()`` yields decoded lines without loading the whole
      file
    * ``str(CONTEXT)`` reads and decodes the entire file (use with care)

    Parameters
    ----------
    path : str | Path
        Path to the file to map.
    encoding : str
        Text encoding used when decoding slices (default ``"utf-8"``).
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._fd = os.open(str(self._path), os.O_RDONLY)
        self._size = os.fstat(self._fd).st_size
        if self._size == 0:
            # mmap cannot map zero-length files.
            self._mm: mmap.mmap | None = None
        else:
            self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the memory map and file descriptor."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:  # noqa: D105
        self.close()

    def __enter__(self) -> LazyContext:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> str: ...

    def __getitem__(self, index: int | slice) -> str:
        if self._mm is None:
            if isinstance(index, slice):
                return ""
            raise IndexError("empty context")
        raw = self._mm[index]
        if isinstance(raw, int):
            return chr(raw)
        return raw.decode(self._encoding, errors="replace")

    def __str__(self) -> str:
        """Decode the entire file.  Use with care for very large files."""
        if self._mm is None:
            return ""
        return self._mm[:].decode(self._encoding, errors="replace")

    def __repr__(self) -> str:
        return f"LazyContext({str(self._path)!r}, size={self._size:,})"

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        if self._mm is None:
            return False
        return self._mm.find(item.encode(self._encoding)) != -1

    # ------------------------------------------------------------------
    # Regex helpers (operate on the raw mmap bytes)
    # ------------------------------------------------------------------

    def search(self, pattern: str, flags: int = 0) -> re.Match[bytes] | None:
        """Search for *pattern* anywhere in the file.

        Parameters
        ----------
        pattern : str
            Regular expression (will be compiled as a ``bytes`` pattern).
        flags : int
            Regex flags (e.g. ``re.IGNORECASE``).

        Returns
        -------
        re.Match[bytes] | None
            Match object (with ``.group()`` returning ``bytes``), or
            ``None``.
        """
        if self._mm is None:
            return None
        return re.search(pattern.encode(self._encoding), self._mm, flags)

    def findall(self, pattern: str, flags: int = 0) -> list[str]:
        """Return all non-overlapping matches of *pattern* as strings.

        Parameters
        ----------
        pattern : str
            Regular expression.
        flags : int
            Regex flags.

        Returns
        -------
        list[str]
            Decoded match strings.
        """
        if self._mm is None:
            return []
        raw_matches = re.findall(pattern.encode(self._encoding), self._mm, flags)
        return [
            m.decode(self._encoding, errors="replace") if isinstance(m, bytes) else str(m)
            for m in raw_matches
        ]

    def lines(self, encoding: str | None = None) -> Iterator[str]:
        """Yield decoded lines without loading the whole file.

        Parameters
        ----------
        encoding : str | None
            Override encoding (defaults to the instance encoding).

        Yields
        ------
        str
            One line at a time (newline stripped).
        """
        enc = encoding or self._encoding
        if self._mm is None:
            return
        # Iterate using readline() which only loads one line at a time.
        self._mm.seek(0)
        while True:
            line = self._mm.readline()
            if not line:
                break
            yield line.decode(enc, errors="replace").rstrip("\n\r")

    def chunk(self, start: int, size: int) -> str:
        """Return a decoded chunk of the file.

        Parameters
        ----------
        start : int
            Byte offset.
        size : int
            Number of bytes to read.

        Returns
        -------
        str
            Decoded text.
        """
        return self[start : start + size]


class StringContext:
    """Thin wrapper around an in-memory ``str`` with the same helper API.

    This lets REPL code use ``CONTEXT.findall(pattern)`` etc. regardless
    of whether the underlying storage is a memory-mapped file or a plain
    string.

    Parameters
    ----------
    text : str
        The full context string.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._text)

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> str: ...

    def __getitem__(self, index: int | slice) -> str:
        return self._text[index]

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"StringContext(size={len(self._text):,})"

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return item in self._text

    # ------------------------------------------------------------------
    # Regex helpers (mirror LazyContext API)
    # ------------------------------------------------------------------

    def search(self, pattern: str, flags: int = 0) -> re.Match[str] | None:
        """Search for *pattern* in the text.

        Parameters
        ----------
        pattern : str
            Regular expression.
        flags : int
            Regex flags.

        Returns
        -------
        re.Match[str] | None
            Match object or ``None``.
        """
        return re.search(pattern, self._text, flags)

    def findall(self, pattern: str, flags: int = 0) -> list[str]:
        """Return all non-overlapping matches of *pattern*.

        Parameters
        ----------
        pattern : str
            Regular expression.
        flags : int
            Regex flags.

        Returns
        -------
        list[str]
            Matched strings.
        """
        return re.findall(pattern, self._text, flags)

    def lines(self, encoding: str | None = None) -> Iterator[str]:
        """Yield lines from the text.

        Parameters
        ----------
        encoding : str | None
            Ignored (present for API compatibility with :class:`LazyContext`).

        Yields
        ------
        str
            One line at a time.
        """
        yield from self._text.splitlines()

    def chunk(self, start: int, size: int) -> str:
        """Return a substring.

        Parameters
        ----------
        start : int
            Character offset.
        size : int
            Number of characters.

        Returns
        -------
        str
            Substring.
        """
        return self._text[start : start + size]

    def close(self) -> None:
        """No-op for in-memory context (nothing to release)."""


# Type alias for any single-file context wrapper.
SingleContext = LazyContext | StringContext

# Separator inserted between files when materialising the concatenated view.
_FILE_SEP = "\n\n"


class CompositeContext:
    """Context backed by multiple files and/or strings.

    Provides:

    * **Per-file access** — ``CONTEXT.files`` lists logical names,
      ``CONTEXT.file("name")`` returns the individual
      :class:`LazyContext`/:class:`StringContext`.
    * **Cross-file search** — ``CONTEXT.findall(pattern)`` returns
      ``list[tuple[str, str]]`` with ``(filename, match)`` pairs.
    * **Concatenated view** — slicing and ``len()`` operate on a virtual
      concatenation (with file-separator markers) for quick sampling.

    Parameters
    ----------
    sources : dict[str, SingleContext]
        Mapping from logical name to context wrapper.
    """

    def __init__(self, sources: dict[str, SingleContext]) -> None:
        self._sources: dict[str, SingleContext] = {}
        if not sources:
            raise ValueError("CompositeContext requires at least one source")
        self._sources = sources

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(cls, paths: list[Path], encoding: str = "utf-8") -> CompositeContext:
        """Create a CompositeContext from a list of file paths.

        Parameters
        ----------
        paths : list[Path]
            Files to include.
        encoding : str
            Text encoding for all files.

        Returns
        -------
        CompositeContext
        """
        sources: dict[str, SingleContext] = {}
        for p in paths:
            key = p.name
            # Disambiguate duplicate basenames by prepending parent dir(s).
            if key in sources:
                key = str(p)
            sources[key] = LazyContext(p, encoding=encoding)
        return cls(sources)

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        glob: str = "**/*",
        encoding: str = "utf-8",
    ) -> CompositeContext:
        """Create a CompositeContext from all files matching *glob* in a directory.

        Parameters
        ----------
        directory : Path
            Root directory.
        glob : str
            Glob pattern (default ``**/*`` — all files recursively).
        encoding : str
            Text encoding.

        Returns
        -------
        CompositeContext
        """
        paths = sorted(p for p in directory.glob(glob) if p.is_file())
        if not paths:
            raise FileNotFoundError(f"No files matched '{glob}' in {directory}")
        sources: dict[str, SingleContext] = {}
        for p in paths:
            key = str(p.relative_to(directory))
            sources[key] = LazyContext(p, encoding=encoding)
        return cls(sources)

    # ------------------------------------------------------------------
    # Per-file access
    # ------------------------------------------------------------------

    @property
    def files(self) -> list[str]:
        """Logical names of all sources, in insertion order."""
        return list(self._sources.keys())

    def file(self, name: str) -> SingleContext:
        """Return the context for a single file.

        Parameters
        ----------
        name : str
            Logical name as listed in :attr:`files`.

        Returns
        -------
        SingleContext

        Raises
        ------
        KeyError
            If *name* is not found.
        """
        return self._sources[name]

    # ------------------------------------------------------------------
    # Concatenated view helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total size across all sources (sum of individual sizes)."""
        return sum(len(s) for s in self._sources.values())

    def __repr__(self) -> str:
        total = len(self)
        n = len(self._sources)
        return f"CompositeContext(files={n}, total_size={total:,})"

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return any(item in s for s in self._sources.values())

    def __str__(self) -> str:
        """Concatenate all sources (use with care for very large contexts)."""
        parts: list[str] = []
        for name, src in self._sources.items():
            parts.extend((f"--- {name} ---", str(src)))
        return _FILE_SEP.join(parts)

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> str: ...

    def __getitem__(self, index: int | slice) -> str:
        """Slice from the concatenated view."""
        # Materialising the full concatenation is expensive for huge
        # contexts; but slicing is typically used on small windows
        # (e.g. CONTEXT[:500]).  We stream through sources to avoid a
        # full copy when possible.
        if isinstance(index, int):
            return str(self)[index]
        return str(self)[index]

    # ------------------------------------------------------------------
    # Cross-file regex helpers
    # ------------------------------------------------------------------

    def search(
        self, pattern: str, flags: int = 0
    ) -> tuple[str, re.Match[str] | re.Match[bytes]] | None:
        """Search across all files; return ``(filename, match)`` for the first hit.

        Parameters
        ----------
        pattern : str
            Regular expression.
        flags : int
            Regex flags.

        Returns
        -------
        tuple[str, Match] | None
        """
        for name, src in self._sources.items():
            m = src.search(pattern, flags)
            if m is not None:
                return (name, m)
        return None

    def findall(self, pattern: str, flags: int = 0) -> list[tuple[str, str]]:
        """Find all matches across all files.

        Parameters
        ----------
        pattern : str
            Regular expression.
        flags : int
            Regex flags.

        Returns
        -------
        list[tuple[str, str]]
            ``(filename, match_text)`` pairs.
        """
        results: list[tuple[str, str]] = []
        for name, src in self._sources.items():
            for match_text in src.findall(pattern, flags):
                results.append((name, match_text))
        return results

    def lines(self, encoding: str | None = None) -> Iterator[tuple[str, str]]:
        """Yield ``(filename, line)`` tuples across all files.

        Parameters
        ----------
        encoding : str | None
            Override encoding.

        Yields
        ------
        tuple[str, str]
        """
        for name, src in self._sources.items():
            for line in src.lines(encoding=encoding):
                yield (name, line)

    def chunk(self, start: int, size: int) -> str:
        """Chunk from the concatenated view.

        Parameters
        ----------
        start : int
            Offset into the concatenated representation.
        size : int
            Number of characters.

        Returns
        -------
        str
        """
        return str(self)[start : start + size]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all underlying contexts."""
        for src in self._sources.values():
            if hasattr(src, "close"):
                src.close()

    def __del__(self) -> None:  # noqa: D105
        self.close()

    def __enter__(self) -> CompositeContext:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
