"""Retrieval evaluation: recall@5, recall@10 and MRR over a question set, at two levels.

- File level: a result counts when its file is one of `expected_paths`. A question with
  `expected_all` (groups of paths) needs a hit in every group; its rank is the position at
  which the last group is first hit.
- Function level: a result counts when it is in the right file and its lines overlap one of
  `expected_spans` ("path:first-last"). Questions without spans score as at file level.
"""
from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config, retrieval

EVAL_DIR = config.PROJECT_ROOT / "eval"
K_MAX = 10
SETS = ("questions", "holdout", "hard", "hard_holdout")


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_span(span: str) -> tuple[str, int, int]:
    path, lines = span.rsplit(":", 1)
    first, last = lines.split("-")
    return path.lower(), int(first), int(last)


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def file_rank(hits: list[Hit], question: dict) -> int | None:
    groups = question.get("expected_all") or [question["expected_paths"]]
    ranks = []
    for group in groups:
        wanted = {_norm(p) for p in group}
        rank = next((i + 1 for i, h in enumerate(hits) if _norm(h.path) in wanted), None)
        if rank is None:
            return None
        ranks.append(rank)
    return max(ranks)


def span_rank(hits: list[Hit], question: dict) -> int | None:
    spans = [parse_span(s) for s in question.get("expected_spans", [])]
    if not spans:
        return file_rank(hits, question)
    for i, h in enumerate(hits):
        if any(_norm(h.path) == p and h.start_line <= last and h.end_line >= first for p, first, last in spans):
            return i + 1
    return None


def score_question(result_paths: list[str], expected: list[str]) -> dict:
    """File-level score for a plain question; result_paths are logical paths, best first."""
    rank = file_rank([Hit(p, 0, 0) for p in result_paths], {"expected_paths": expected})
    return _from_rank(rank)


def _from_rank(rank: int | None) -> dict:
    return {
        "recall@5": float(rank is not None and rank <= 5),
        "recall@10": float(rank is not None and rank <= 10),
        "mrr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def run_mode(
    conn: sqlite3.Connection, questions: list[dict], mode: str, model: str | None = None, reranker: str | None = None
) -> dict:
    """Averages over the set, plus `fn_*` function-level metrics, misses and mean latency."""
    retrieval.search(conn, "warm-up", k=K_MAX, mode=mode, model=model, reranker=reranker)  # loading is not latency
    file_scores, fn_scores, seconds = [], [], []
    for q in questions:
        t0 = time.perf_counter()
        results = retrieval.search(conn, q["question"], k=K_MAX, mode=mode, model=model, reranker=reranker)
        seconds.append(time.perf_counter() - t0)
        hits = [Hit(r.path, r.start_line, r.end_line) for r in results]
        file_scores.append(_from_rank(file_rank(hits, q)))
        fn_scores.append(_from_rank(span_rank(hits, q)))
    n = len(questions) or 1
    out = {key: sum(s[key] for s in file_scores) / n for key in ("recall@5", "recall@10", "mrr")}
    out.update({f"fn_{key}": sum(s[key] for s in fn_scores) / n for key in ("recall@5", "recall@10", "mrr")})
    out["misses"] = [q["id"] for q, s in zip(questions, file_scores) if s["rank"] is None or s["rank"] > 5]
    out["fn_misses"] = [q["id"] for q, s in zip(questions, fn_scores) if s["rank"] is None or s["rank"] > 5]
    out["latency_ms"] = 1000 * statistics.mean(seconds) if seconds else 0.0
    return out
