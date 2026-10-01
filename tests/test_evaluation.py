"""Scoring at file level and at function level, and the shape of the question files."""
import pytest
import yaml

from code_rag.evaluation import EVAL_DIR, SETS, Hit, file_rank, parse_span, score_question, span_rank

Q = {
    "expected_paths": ["r/a.py", "r/b.py"],
    "expected_spans": ["r/a.py:100-120"],
}


def test_file_rank_is_the_first_expected_file():
    hits = [Hit("r/x.py", 1, 9), Hit("r/b.py", 1, 9), Hit("r/a.py", 1, 9)]
    assert file_rank(hits, Q) == 2


def test_span_rank_needs_overlapping_lines_in_the_right_file():
    hits = [Hit("r/a.py", 1, 50), Hit("r/b.py", 100, 120), Hit("r/a.py", 110, 130)]
    assert span_rank(hits, Q) == 3


def test_span_rank_counts_a_chunk_that_contains_the_whole_span():
    assert span_rank([Hit("r/a.py", 90, 200)], Q) == 1


def test_question_without_spans_scores_at_file_level():
    q = {"expected_paths": ["r/a.py"]}
    assert span_rank([Hit("r/x.py", 1, 2), Hit("r/a.py", 1, 2)], q) == 2


def test_all_groups_must_be_hit_and_the_rank_is_the_last_one():
    q = {"expected_paths": ["r/a.py", "r/b.py", "r/c.py"], "expected_all": [["r/a.py"], ["r/b.py", "r/c.py"]]}
    assert file_rank([Hit("r/c.py", 1, 2), Hit("r/x.py", 1, 2), Hit("r/a.py", 1, 2)], q) == 3
    assert file_rank([Hit("r/c.py", 1, 2), Hit("r/b.py", 1, 2)], q) is None


def test_plain_score_question_is_unchanged():
    s = score_question(["r/x.py", "r/A.py"], ["r/a.py"])
    assert (s["rank"], s["recall@5"], s["mrr"]) == (2, 1.0, 0.5)


def test_parse_span():
    assert parse_span("httpx/httpx/_client.py:546-571") == ("httpx/httpx/_client.py", 546, 571)


@pytest.mark.parametrize("name", SETS)
def test_question_files_are_well_formed(name):
    questions = yaml.safe_load((EVAL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    for q in questions:
        assert q["id"] and q["question"] and q["expected_paths"]
        for span in q.get("expected_spans", []):
            path, first, last = parse_span(span)
            assert path in {p.lower() for p in q["expected_paths"]} and 1 <= first <= last
        for group in q.get("expected_all", []):
            assert set(group) <= set(q["expected_paths"])


def test_question_ids_are_unique_across_sets():
    ids = [q["id"] for name in SETS for q in yaml.safe_load((EVAL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))]
    assert len(ids) == len(set(ids))
