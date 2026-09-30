"""Retrieval: BM25 (FTS5), dense cosine (numpy), and hybrid RRF fusion.

Every mode applies the same per-file cap (config.MAX_CHUNKS_PER_FILE) to
the final top-k: candidates are pulled from a larger internal pool, then
capped so one file's tests, docs and other chunks cannot crowd out the
single best chunk of every other file.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config, store


@dataclass
class SearchResult:
    chunk_id: int
    path: str
    repo: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    snippet: str
    score: float


def _pool_size(k: int) -> int:
    return max(k * config.RETRIEVAL_POOL_MULTIPLIER, config.RETRIEVAL_MIN_POOL)


_NON_SOURCE = re.compile(
    r"(^|/)(tests?|docs?|examples?|benchmarks?|questions)/|(^|/)test_[^/]*\.py$|_test\.py$"
    r"|(^|/)(CHANGELOG|CHANGES|HISTORY|FAQ|README|CONTRIBUTING)[^/]*$|\.(md|rst|txt)$", re.I)


def source_prior(path: str) -> float:
    """Tests, docs and changelogs match questions well but rarely hold the implementation an agent asks for."""
    return config.NON_SOURCE_WEIGHT if _NON_SOURCE.search(path) else 1.0


def _apply_prior(scored_ids: list[tuple[int, float]], id_to_path: dict[int, str]) -> list[tuple[int, float]]:
    """Scores must be higher-is-better here."""
    if config.NON_SOURCE_WEIGHT == 1.0:
        return scored_ids
    weighted = [(cid, score * source_prior(id_to_path.get(cid, ""))) for cid, score in scored_ids]
    return sorted(weighted, key=lambda item: -item[1])


def _cap_per_file(
    scored_ids: list[tuple[int, float]], id_to_path: dict[int, str], cap: int, k: int
) -> list[tuple[int, float]]:
    counts: dict[str, int] = {}
    out: list[tuple[int, float]] = []
    for cid, score in scored_ids:
        path = id_to_path.get(cid)
        if path is None:
            continue
        n = counts.get(path, 0)
        if n >= cap:
            continue
        counts[path] = n + 1
        out.append((cid, score))
        if len(out) >= k:
            break
    return out


def _snippet(text: str, max_chars: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0].strip() + " ..."


def _to_results(conn: sqlite3.Connection, scored_ids: list[tuple[int, float]]) -> list[SearchResult]:
    results = []
    for cid, score in scored_ids:
        chunk = store.get_chunk(conn, cid)
        if chunk is None:
            continue
        results.append(
            SearchResult(
                chunk_id=cid,
                path=chunk["path"],
                repo=chunk["repo"],
                symbol=chunk["symbol"],
                kind=chunk["kind"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                snippet=_snippet(chunk["text"]),
                score=score,
            )
        )
    return results


def bm25_search(conn: sqlite3.Connection, query: str, k: int, repo: "Optional[str]" = None) -> list[SearchResult]:
    # FTS5 bm25() is lower-is-better; negate it so every mode ranks higher-is-better.
    pool = [(cid, -score) for cid, score in store.bm25_search(conn, query, _pool_size(k), repo)]
    id_to_path = store.get_paths_for_ids(conn, [cid for cid, _ in pool])
    capped = _cap_per_file(_apply_prior(pool, id_to_path), id_to_path, config.MAX_CHUNKS_PER_FILE, k)
    return _to_results(conn, capped)


def dense_search_ids(
    conn: sqlite3.Connection, model: str, query: str, k: int, repo: "Optional[str]" = None
) -> list[tuple[int, float]]:
    from . import embeddings as emb

    ids, mat = store.load_embeddings(conn, model)
    if ids.size == 0:
        return []
    if repo:
        allowed = store.chunk_ids_for_repo(conn, repo)
        mask = np.array([i in allowed for i in ids])
        ids, mat = ids[mask], mat[mask]
    if ids.size == 0:
        return []
    qvec = emb.embed_query(model, query)
    scores = mat @ qvec
    top = min(k, scores.shape[0])
    order = np.argpartition(-scores, top - 1)[:top]
    order = order[np.argsort(-scores[order])]
    return [(int(ids[i]), float(scores[i])) for i in order]


def dense_search(
    conn: sqlite3.Connection, model: str, query: str, k: int, repo: "Optional[str]" = None
) -> list[SearchResult]:
    pool = dense_search_ids(conn, model, query, _pool_size(k), repo)
    id_to_path = store.get_paths_for_ids(conn, [cid for cid, _ in pool])
    capped = _cap_per_file(_apply_prior(pool, id_to_path), id_to_path, config.MAX_CHUNKS_PER_FILE, k)
    return _to_results(conn, capped)


def rrf_fuse(rank_lists: "list[list[int]]", k: int = 60, weights: "list[float] | None" = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Each rank_list is a list of chunk ids, best first."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for ranks, weight in zip(rank_lists, weights):
        for rank, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(
    conn: sqlite3.Connection,
    model: str,
    query: str,
    k: int,
    repo: "Optional[str]" = None,
) -> list[SearchResult]:
    pool = _pool_size(k)
    bm25_ids = [cid for cid, _ in store.bm25_search(conn, query, pool, repo)]
    dense_ids = [cid for cid, _ in dense_search_ids(conn, model, query, pool, repo)]
    fused = rrf_fuse([bm25_ids, dense_ids], k=config.RRF_K, weights=[config.BM25_RRF_WEIGHT, 1.0])
    id_to_path = store.get_paths_for_ids(conn, [cid for cid, _ in fused])
    capped = _cap_per_file(_apply_prior(fused, id_to_path), id_to_path, config.MAX_CHUNKS_PER_FILE, k)
    return _to_results(conn, capped)


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 8,
    mode: str = "hybrid",
    repo: "Optional[str]" = None,
    model: "Optional[str]" = None,
) -> list[SearchResult]:
    model = model or config.DEFAULT_MODEL
    if mode == "bm25":
        return bm25_search(conn, query, k, repo)
    if mode == "dense":
        return dense_search(conn, model, query, k, repo)
    if mode == "hybrid":
        return hybrid_search(conn, model, query, k, repo)
    raise ValueError(f"unknown mode: {mode}")
