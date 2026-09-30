from code_rag.chunker import chunk_python, chunk_markdown, chunk_generic, chunk_text, MAX_CHUNK_CHARS

SAMPLE_PY = '''"""Module docstring for the sample."""
from __future__ import annotations

import os
import sys


def top_level_function(x, y):
    """Adds two numbers."""
    return x + y


class SmallClass:
    """A small class, should not be split."""

    def method_one(self):
        return 1

    def method_two(self):
        return 2


class BigClass:
    """A class with many methods, should be split by method."""

    def m1(self):
        return 1

    def m2(self):
        return 2

    def m3(self):
        return 3

    def m4(self):
        return 4

    def m5(self):
        return 5

    def m6(self):
        return 6

    def m7(self):
        return 7
'''


def test_module_chunk_has_docstring_and_imports():
    chunks = chunk_python(SAMPLE_PY, "sample.py", "demo")
    module_chunks = [c for c in chunks if c.kind == "module"]
    assert len(module_chunks) == 1
    assert "Module docstring" in module_chunks[0].text
    assert "import os" in module_chunks[0].text
    assert module_chunks[0].start_line == 1


def test_top_level_function_is_its_own_chunk():
    chunks = chunk_python(SAMPLE_PY, "sample.py", "demo")
    fn_chunks = [c for c in chunks if c.kind == "function" and c.symbol == "top_level_function"]
    assert len(fn_chunks) == 1
    assert "return x + y" in fn_chunks[0].text


def test_small_class_is_a_single_chunk():
    chunks = chunk_python(SAMPLE_PY, "sample.py", "demo")
    class_chunks = [c for c in chunks if c.symbol == "SmallClass"]
    assert len(class_chunks) == 1
    assert class_chunks[0].kind == "class"
    assert "method_one" in class_chunks[0].text
    assert "method_two" in class_chunks[0].text


def test_big_class_is_split_by_method():
    chunks = chunk_python(SAMPLE_PY, "sample.py", "demo")
    method_chunks = [c for c in chunks if c.symbol.startswith("BigClass.")]
    assert len(method_chunks) == 7
    for c in method_chunks:
        assert c.kind == "method"
    header_chunks = [c for c in chunks if c.symbol == "BigClass"]
    assert len(header_chunks) == 1
    assert header_chunks[0].kind == "class"
    assert "def m1" not in header_chunks[0].text


def test_all_chunks_carry_path_and_repo():
    chunks = chunk_python(SAMPLE_PY, "sample.py", "demo")
    for c in chunks:
        assert c.path == "sample.py"
        assert c.repo == "demo"
        assert c.start_line <= c.end_line


def test_chunk_python_handles_syntax_error_gracefully():
    broken = "def f(:\n    pass\n"
    chunks = chunk_python(broken, "broken.py", "demo")
    # Falls back to window chunking instead of raising.
    assert len(chunks) >= 1


MARKDOWN_SAMPLE = """# Title

Intro text.

## Section A

Content A.

## Section B

Content B.
"""


def test_markdown_splits_by_heading():
    chunks = chunk_markdown(MARKDOWN_SAMPLE, "doc.md", "demo")
    titles = [c.symbol for c in chunks]
    assert "Title" in titles
    assert "Section A" in titles
    assert "Section B" in titles
    section_a = next(c for c in chunks if c.symbol == "Section A")
    assert "Content A" in section_a.text
    assert "Content B" not in section_a.text


def test_oversized_top_level_chunk_is_split_into_windows():
    # A huge top-level dict literal (no functions around it) used to become
    # one enormous "module" gap-fill chunk — bigger than what any embedding
    # model actually looks at. chunk_text() must split it further.
    body_lines = [f'    "key_{i}": "value_{i}" * 5,' for i in range(2000)]
    huge_source = "BIG = {\n" + "\n".join(body_lines) + "\n}\n"
    assert len(huge_source) > MAX_CHUNK_CHARS

    chunks = chunk_text(huge_source, "big.py", "demo", ".py")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    # Still fully covers the file's lines, just spread over more chunks.
    total_lines = len(huge_source.splitlines())
    covered = set()
    for c in chunks:
        for ln in range(c.start_line, c.end_line + 1):
            covered.add(ln)
    assert covered == set(range(1, total_lines + 1))


def test_pathologically_long_single_line_falls_back_to_char_split():
    huge_line = "x = '" + ("a" * 10000) + "'\n"
    chunks = chunk_text(huge_line, "oneline.py", "demo", ".py")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_normal_sized_chunks_are_left_alone():
    chunks = chunk_text(SAMPLE_PY, "sample.py", "demo", ".py")
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS
    # No spurious splitting: chunk count matches the un-split chunker.
    assert len(chunks) == len(chunk_python(SAMPLE_PY, "sample.py", "demo"))


def test_generic_chunker_windows_with_overlap():
    lines = [f"line {i}" for i in range(1, 151)]
    text = "\n".join(lines)
    chunks = chunk_generic(text, "file.txt", "demo")
    assert len(chunks) > 1
    # Every line should appear in at least one chunk.
    covered = set()
    for c in chunks:
        for ln in range(c.start_line, c.end_line + 1):
            covered.add(ln)
    assert covered == set(range(1, 151))
