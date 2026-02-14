"""Lazy, memory-mapped context for RLM.

Provides a :class:`LazyContext` wrapper that memory-maps a file so the OS
pages data in and out on demand.  This allows RLM to work with documents
that exceed available process memory — the 64-bit virtual address space
is the only hard limit.

For callers that already have the context as a ``str``, a thin
:class:`StringContext` wrapper exposes the same helper API so REPL code
can use ``CONTEXT.search()`` / ``CONTEXT.findall()`` regardless of the
underlying storage.
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
            yield line.decode(enc, errors="replace").rstrip("\n").rstrip("\r")

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
