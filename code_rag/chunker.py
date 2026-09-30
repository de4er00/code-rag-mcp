"""Splits file contents into retrievable chunks.

- Python: AST-based — one chunk per top-level function, one per class (or,
  for big classes, one header chunk + one chunk per method), and one
  module-level chunk holding the docstring and imports.
- Markdown: one chunk per heading section.
- Everything else: ~60-line windows with a small overlap.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

WINDOW_SIZE = 60
WINDOW_OVERLAP = 10
BIG_CLASS_LINES = 80
BIG_CLASS_METHODS = 6

# Embedding models here truncate at 512 tokens (~a few thousand characters).
# Any chunk bigger than this — a big top-level data table, a giant prompt
# string constant, a huge gap-fill "module" chunk — gets split further so
# nothing past the truncation point is silently unsearchable.
MAX_CHUNK_CHARS = 4000
_CHAR_SPLIT_OVERLAP = 200


@dataclass
class Chunk:
    path: str
    repo: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    text: str


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _decorated_start(node) -> int:
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return min(d.lineno for d in decorators)
    return node.lineno


def _slice(lines: list[str], start_line: int, end_line: int) -> str:
    start_line = max(1, start_line)
    end_line = max(start_line, end_line)
    return "\n".join(lines[start_line - 1 : end_line])


def chunk_python(text: str, path: str, repo: str) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_generic(text, path, repo)

    chunks: list[Chunk] = []
    top_nodes = tree.body

    # Module-level chunk: docstring + top-level imports.
    last_line = 0
    if (
        top_nodes
        and isinstance(top_nodes[0], ast.Expr)
        and isinstance(getattr(top_nodes[0], "value", None), (ast.Constant,))
        and isinstance(top_nodes[0].value.value, str)
    ):
        last_line = max(last_line, top_nodes[0].end_lineno)
    for n in top_nodes:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last_line = max(last_line, n.end_lineno)
    if last_line > 0:
        chunks.append(
            Chunk(path, repo, "<module>", "module", 1, last_line, _slice(lines, 1, last_line))
        )

    for node in top_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _decorated_start(node), node.end_lineno
            chunks.append(
                Chunk(path, repo, node.name, "function", start, end, _slice(lines, start, end))
            )
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            class_start, class_end = _decorated_start(node), node.end_lineno
            is_big = methods and (
                (class_end - class_start) > BIG_CLASS_LINES or len(methods) > BIG_CLASS_METHODS
            )
            if not is_big:
                chunks.append(
                    Chunk(
                        path, repo, node.name, "class", class_start, class_end,
                        _slice(lines, class_start, class_end),
                    )
                )
            else:
                first_method_start = _decorated_start(methods[0])
                header_end = first_method_start - 1
                if header_end >= class_start:
                    chunks.append(
                        Chunk(
                            path, repo, node.name, "class", class_start, header_end,
                            _slice(lines, class_start, header_end),
                        )
                    )
                for m in methods:
                    m_start, m_end = _decorated_start(m), m.end_lineno
                    chunks.append(
                        Chunk(
                            path, repo, f"{node.name}.{m.name}", "method", m_start, m_end,
                            _slice(lines, m_start, m_end),
                        )
                    )

    # Top-level statements that are neither part of the module
    # docstring/imports nor a top-level def/class (constants, argparse
    # setup, `if __name__ == "__main__":` blocks, bare script code) would
    # otherwise be silently dropped. Fill the line-number gaps left by the
    # chunks above with extra "module" chunks so every line — including a
    # secret-looking assignment sitting in a bare script — passes through
    # redaction and is not lost from the index.
    covered: list[tuple[int, int]] = [(c.start_line, c.end_line) for c in chunks]
    covered.sort()
    merged: list[list[int]] = []
    for s, e in covered:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    gap_start = 1
    gaps: list[tuple[int, int]] = []
    for s, e in merged:
        if gap_start < s:
            gaps.append((gap_start, s - 1))
        gap_start = max(gap_start, e + 1)
    if gap_start <= len(lines):
        gaps.append((gap_start, len(lines)))
    for g_start, g_end in gaps:
        snippet = _slice(lines, g_start, g_end)
        if snippet.strip():
            chunks.append(Chunk(path, repo, "<module>", "module", g_start, g_end, snippet))

    if not chunks:
        return chunk_generic(text, path, repo)
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(text: str, path: str, repo: str) -> list[Chunk]:
    lines = text.splitlines()
    heading_idx = [i for i, l in enumerate(lines) if _HEADING_RE.match(l)]
    if not heading_idx:
        return chunk_generic(text, path, repo)

    chunks: list[Chunk] = []
    if heading_idx[0] > 0:
        chunks.append(
            Chunk(path, repo, "<preamble>", "markdown_section", 1, heading_idx[0], "\n".join(lines[: heading_idx[0]]))
        )
    for i, start_idx in enumerate(heading_idx):
        end_idx = heading_idx[i + 1] if i + 1 < len(heading_idx) else len(lines)
        title = _HEADING_RE.match(lines[start_idx]).group(2).strip()
        start_line, end_line = start_idx + 1, end_idx
        chunks.append(
            Chunk(path, repo, title or "<section>", "markdown_section", start_line, end_line, "\n".join(lines[start_idx:end_idx]))
        )
    return chunks


def chunk_generic(text: str, path: str, repo: str) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[Chunk] = []
    step = WINDOW_SIZE - WINDOW_OVERLAP
    start = 0
    n = len(lines)
    while start < n:
        end = min(start + WINDOW_SIZE, n)
        chunk_lines = lines[start:end]
        chunks.append(
            Chunk(
                path, repo, f"lines {start + 1}-{end}", "window", start + 1, end,
                "\n".join(chunk_lines),
            )
        )
        if end == n:
            break
        start += step
    return chunks


def _char_split(c: Chunk) -> list[Chunk]:
    """Last-resort split for a chunk that is still too big after line-window
    splitting (e.g. one pathologically long line)."""
    text = c.text
    n = len(text)
    step = MAX_CHUNK_CHARS - _CHAR_SPLIT_OVERLAP
    parts: list[Chunk] = []
    i = 0
    part_num = 0
    while i < n:
        j = min(i + MAX_CHUNK_CHARS, n)
        part_num += 1
        parts.append(
            Chunk(c.path, c.repo, f"{c.symbol} [chars {i}-{j}]", c.kind, c.start_line, c.end_line, text[i:j])
        )
        if j == n:
            break
        i += step
    return parts


def _split_oversized(c: Chunk) -> list[Chunk]:
    """Split a chunk bigger than MAX_CHUNK_CHARS into overlapping line
    windows (falling back to a raw character split if a single line window
    is itself still too big)."""
    if len(c.text) <= MAX_CHUNK_CHARS:
        return [c]

    lines = c.text.splitlines()
    if len(lines) <= 1:
        return _char_split(c)

    parts: list[Chunk] = []
    step = WINDOW_SIZE - WINDOW_OVERLAP
    n = len(lines)
    idx = 0
    part_num = 0
    while idx < n:
        end_idx = min(idx + WINDOW_SIZE, n)
        sub_text = "\n".join(lines[idx:end_idx])
        sub_start = c.start_line + idx
        sub_end = c.start_line + end_idx - 1
        part_num += 1
        sub = Chunk(c.path, c.repo, f"{c.symbol} [part {part_num}]", c.kind, sub_start, sub_end, sub_text)
        if len(sub_text) > MAX_CHUNK_CHARS:
            parts.extend(_char_split(sub))
        else:
            parts.append(sub)
        if end_idx == n:
            break
        idx += step
    return parts


def chunk_text(text: str, path: str, repo: str, suffix: str) -> list[Chunk]:
    suffix = suffix.lower()
    if suffix == ".py":
        raw = chunk_python(text, path, repo)
    elif suffix == ".md":
        raw = chunk_markdown(text, path, repo)
    else:
        raw = chunk_generic(text, path, repo)

    out: list[Chunk] = []
    for c in raw:
        out.extend(_split_oversized(c))
    return out


def chunk_file(path: Path, repo: str) -> list[Chunk]:
    text = read_text(path)
    return chunk_text(text, str(path), repo, path.suffix)
