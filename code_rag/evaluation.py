"""Retrieval evaluation: recall@5, recall@10 and MRR at file level over a question set."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from . import config, retrieval

EVAL_DIR = config.PROJECT_ROOT / "eval"
K_MAX = 10


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def score_question(result_paths: list[str], expected: list[str]) -> dict:
    """result_paths are logical paths ('repo/relative/path'), best first."""
    wanted = {e.lower() for e in expected}
    rank = next((i + 1 for i, p in enumerate(result_paths) if p.replace("\\", "/").lower() in wanted), None)
    return {
        "recall@5": float(rank is not None and rank <= 5),
        "recall@10": float(rank is not None and rank <= 10),
        "mrr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def run_mode(conn: sqlite3.Connection, questions: list[dict], mode: str, model: str | None = None) -> dict:
    totals = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    misses = []
    for q in questions:
        paths = [r.path for r in retrieval.search(conn, q["question"], k=K_MAX, mode=mode, model=model)]
        s = score_question(paths, q["expected_paths"])
        for key in totals:
            totals[key] += s[key]
        if s["rank"] is None or s["rank"] > 5:
            misses.append(q["id"])
    n = len(questions) or 1
    return {**{k: v / n for k, v in totals.items()}, "misses": misses}
