"""SQLite storage: file hashes, chunks, FTS5 keyword index, embedding blobs.

Each chunk carries a contextual header (path | symbol | kind). The keyword
index has two FTS columns: `header_tokens` (path, symbol and kind split on
separators and camelCase) and `body_stemmed` (stopword-filtered, stemmed
text), so BM25 can weight an exact filename or symbol match above a match
in the body. See code_rag/textindex.py.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from . import config, textindex
from .chunker import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    repo TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    header TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(header_tokens, body_stemmed, tokenize='unicode61');

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"""


def connect(db_path: "Path | None" = None) -> sqlite3.Connection:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_known_files(conn: sqlite3.Connection) -> dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("SELECT path, sha256 FROM files")}


def upsert_file(conn: sqlite3.Connection, path: str, sha256: str, size: int, mtime: float) -> None:
    conn.execute(
        "INSERT INTO files(path, sha256, size, mtime) VALUES (?,?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, size=excluded.size, mtime=excluded.mtime",
        (path, sha256, size, mtime),
    )


def delete_file(conn: sqlite3.Connection, path: str) -> int:
    """Remove a file's row, its chunks, their FTS entries and embeddings. Returns chunks removed."""
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE path=?", (path,))]
    for cid in ids:
        conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
    if ids:
        conn.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({','.join('?' * len(ids))})", ids)
    conn.execute("DELETE FROM chunks WHERE path=?", (path,))
    conn.execute("DELETE FROM files WHERE path=?", (path,))
    return len(ids)


def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> list[int]:
    ids = []
    for c in chunks:
        header = textindex.make_header(c.path, c.symbol, c.kind)
        cur = conn.execute(
            "INSERT INTO chunks(path, repo, symbol, kind, start_line, end_line, text, header) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (c.path, c.repo, c.symbol, c.kind, c.start_line, c.end_line, c.text, header),
        )
        cid = cur.lastrowid
        header_tokens = textindex.header_tokens_text(c.path, c.repo, c.symbol, c.kind)
        body_stemmed = textindex.stem_text(c.text)
        conn.execute(
            "INSERT INTO chunks_fts(rowid, header_tokens, body_stemmed) VALUES (?, ?, ?)",
            (cid, header_tokens, body_stemmed),
        )
        ids.append(cid)
    return ids


def chunk_ids_without_embedding(conn: sqlite3.Connection, model: str) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM chunks c LEFT JOIN embeddings e "
        "ON e.chunk_id = c.id AND e.model = ? WHERE e.chunk_id IS NULL",
        (model,),
    )
    return [r[0] for r in rows]


def insert_embeddings(conn: sqlite3.Connection, model: str, chunk_ids: list[int], vectors: np.ndarray) -> None:
    dim = int(vectors.shape[1])
    rows = [
        (cid, model, dim, np.asarray(vectors[i], dtype=np.float32).tobytes())
        for i, cid in enumerate(chunk_ids)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?,?,?,?)", rows
    )


def load_embeddings(conn: sqlite3.Connection, model: str) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT chunk_id, dim, vector FROM embeddings WHERE model=? ORDER BY chunk_id", (model,)
    ).fetchall()
    if not rows:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    dim = rows[0][1]
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]).reshape(len(rows), dim)
    return ids, mat


def get_chunk(conn: sqlite3.Connection, chunk_id: int) -> "Optional[dict]":
    row = conn.execute(
        "SELECT id, path, repo, symbol, kind, start_line, end_line, text, header "
        "FROM chunks WHERE id=?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    keys = ["id", "path", "repo", "symbol", "kind", "start_line", "end_line", "text", "header"]
    return dict(zip(keys, row))


def list_repos(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    return list(conn.execute("SELECT repo, COUNT(*) FROM chunks GROUP BY repo ORDER BY repo"))


def bm25_search(
    conn: sqlite3.Connection, query: str, k: int, repo: "Optional[str]" = None
) -> list[tuple[int, float]]:
    """BM25 over two FTS columns: `header_tokens` (path/symbol/kind, exact
    tokens) and `body_stemmed` (stemmed body text). Header matches are
    weighted higher — an exact filename/symbol hit is a much stronger
    signal than one stemmed word matching somewhere in the body."""
    raw_tokens = textindex.query_tokens(query)
    if not raw_tokens:
        return []
    stemmed_tokens = [textindex.stem_token(t) for t in raw_tokens]

    header_clause = " OR ".join(f'header_tokens:"{t}"' for t in raw_tokens)
    body_clause = " OR ".join(f'body_stemmed:"{t}"' for t in stemmed_tokens)
    fts_query = f"({header_clause}) OR ({body_clause})"

    sql = (
        f"SELECT c.id, bm25(chunks_fts, {config.HEADER_BM25_WEIGHT}, {config.BODY_BM25_WEIGHT}) as score "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ?"
    )
    params: list = [fts_query]
    if repo:
        sql += " AND c.repo = ?"
        params.append(repo)
    sql += " ORDER BY score LIMIT ?"
    params.append(k)
    return [(r[0], r[1]) for r in conn.execute(sql, params)]


def chunk_ids_for_repo(conn: sqlite3.Connection, repo: str) -> set[int]:
    return {r[0] for r in conn.execute("SELECT id FROM chunks WHERE repo=?", (repo,))}


def get_texts_for_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", batch)
        result.update(rows)
    return result


def get_embedding_inputs_for_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    """header + chunk text, concatenated — this is what actually gets
    embedded, so the vector carries the path/symbol/kind signal too."""
    result: dict[int, str] = {}
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(f"SELECT id, header, text FROM chunks WHERE id IN ({placeholders})", batch)
        for cid, header, text in rows:
            result[cid] = f"{header}\n{text}" if header else text
    return result


def get_paths_for_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(f"SELECT id, path FROM chunks WHERE id IN ({placeholders})", batch)
        result.update(rows)
    return result


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
