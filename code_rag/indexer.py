"""Walk -> chunk -> redact -> store -> embed, incrementally.

A file whose sha256 did not change is neither re-chunked nor re-embedded; a file that disappeared
loses its chunks, keyword entries and vectors. Embedding is committed in small batches, so an
interrupted run resumes from the chunks that still have no vector.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import config, store
from .chunker import chunk_file
from .redact import redact_text
from .walker import print_summary, walk_source

COMMIT_BATCH_SIZE = 64
PROGRESS_INTERVAL_S = 5.0


@dataclass
class IndexReport:
    files_included: int = 0
    files_unchanged: int = 0
    files_reindexed: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_total: int = 0
    redaction_counts: Counter = field(default_factory=Counter)
    embedding_seconds: dict = field(default_factory=dict)
    embedding_chunks_embedded: dict = field(default_factory=dict)


def _index_files(conn, repo: config.Repo, report: IndexReport, known: dict[str, str], found: set[str],
                 verbose: bool) -> None:
    walk = walk_source(repo.root)
    report.files_included += len(walk.included)
    if verbose:
        print_summary(repo.name, walk)
    for path in walk.included:
        # The index stores a logical path, never an absolute one: the database stays portable and
        # carries no local directory names.
        spath = f"{repo.name}/{path.relative_to(repo.root).as_posix()}"
        found.add(spath)
        try:
            sha = store.file_sha256(path)
        except OSError as exc:
            if verbose:
                print(f"  ! could not read {spath}: {exc}")
            continue
        if known.get(spath) == sha:
            report.files_unchanged += 1
            continue
        if spath in known:
            store.delete_file(conn, spath)
        try:
            chunks = chunk_file(path, repo.name)
        except Exception as exc:  # a file that fails to parse must not stop the whole index
            if verbose:
                print(f"  ! failed to chunk {spath}: {exc}")
            continue
        for chunk in chunks:
            chunk.path = spath
            chunk.text, counts = redact_text(chunk.text)
            report.redaction_counts.update(counts)
        report.chunks_added += len(store.insert_chunks(conn, chunks))
        report.files_reindexed += 1
        stat = path.stat()
        store.upsert_file(conn, spath, sha, stat.st_size, stat.st_mtime)


def _embed_missing(conn, model: str, report: IndexReport, verbose: bool) -> None:
    from . import embeddings as emb

    missing = store.chunk_ids_without_embedding(conn, model)
    started, last, done = time.time(), time.time(), 0
    if missing and verbose:
        print(f"Embedding {len(missing)} chunks with {model} ...", flush=True)
    for i in range(0, len(missing), COMMIT_BATCH_SIZE):
        texts = store.get_embedding_inputs_for_ids(conn, missing[i:i + COMMIT_BATCH_SIZE])
        ids = list(texts)
        if not ids:
            continue
        store.insert_embeddings(conn, model, ids, emb.embed_passages(model, [texts[c] for c in ids]))
        conn.commit()
        done += len(ids)
        now = time.time()
        if verbose and (now - last >= PROGRESS_INTERVAL_S or done >= len(missing)):
            rate = done / max(now - started, 1e-9)
            print(f"  [{model}] {done}/{len(missing)} chunks, {rate:.1f}/s", flush=True)
            last = now
    report.embedding_seconds[model] = time.time() - started
    report.embedding_chunks_embedded[model] = done
    emb.unload_model(model)


def index_corpus(repos: list[config.Repo], models: list[str] | None = None, db_path: Path | None = None,
                 verbose: bool = True) -> IndexReport:
    models = models if models is not None else [config.DEFAULT_MODEL]
    conn = store.connect(db_path)
    store.init_db(conn)
    known = store.get_known_files(conn)
    found: set[str] = set()
    report = IndexReport()
    for repo in repos:
        _index_files(conn, repo, report, known, found, verbose)
    gone = [p for p in known if p not in found]
    for path in gone:
        store.delete_file(conn, path)
    report.files_deleted = len(gone)
    conn.commit()
    report.chunks_total = store.count_chunks(conn)
    for model in models:
        _embed_missing(conn, model, report, verbose)
    conn.close()
    return report


def print_report(report: IndexReport) -> None:
    print("\n--- Index report ---")
    print(f"Files included: {report.files_included} (unchanged {report.files_unchanged}, "
          f"re-indexed {report.files_reindexed}, deleted {report.files_deleted})")
    print(f"Chunks added: {report.chunks_added}, total: {report.chunks_total}")
    if report.redaction_counts:
        print("Redactions: " + ", ".join(f"{k} {v}" for k, v in report.redaction_counts.most_common()))
    for model, secs in report.embedding_seconds.items():
        print(f"Embedded with {model}: {report.embedding_chunks_embedded.get(model, 0)} chunks in {secs:.1f}s")
