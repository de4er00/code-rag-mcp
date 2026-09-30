"""Incremental re-index: unchanged files are not re-embedded, deleted files
are removed from chunks/FTS/embeddings. Uses a fake embedding function so no
model download happens during tests."""
import numpy as np
import pytest

from code_rag import store
from code_rag.config import Repo
from code_rag.indexer import index_corpus

FAKE_MODEL = "fake-test-model"


def _fake_embed_passages(model, texts):
    # Deterministic tiny vectors, just enough to exercise the storage path.
    return np.array([[float(len(t) % 7), 1.0, 0.0] for t in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    monkeypatch.setattr("code_rag.embeddings.embed_passages", _fake_embed_passages)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "demo-project"
    root.mkdir(parents=True)
    (root / "a.py").write_text('"""a module."""\n\n\ndef f():\n    return 1\n', encoding="utf-8")
    db_path = tmp_path / "index.db"
    return root, db_path


def test_first_index_embeds_everything(project):
    root, db_path = project
    report = index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)
    assert report.files_reindexed == 1
    assert report.files_unchanged == 0
    assert report.chunks_added > 0
    assert report.embedding_chunks_embedded[FAKE_MODEL] == report.chunks_added


def test_unchanged_file_is_not_reembedded(project):
    root, db_path = project
    index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)

    report2 = index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)
    assert report2.files_unchanged == 1
    assert report2.files_reindexed == 0
    assert report2.chunks_added == 0
    assert report2.embedding_chunks_embedded[FAKE_MODEL] == 0


def test_changed_file_is_rechunked_and_reembedded(project):
    root, db_path = project
    index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)

    a_py = root / "a.py"
    a_py.write_text(
        '"""a module, changed."""\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n',
        encoding="utf-8",
    )
    report3 = index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)
    assert report3.files_reindexed == 1
    assert report3.files_unchanged == 0
    assert report3.chunks_added > 0
    assert report3.embedding_chunks_embedded[FAKE_MODEL] == report3.chunks_added

    conn = store.connect(db_path)
    texts = [row[0] for row in conn.execute("SELECT text FROM chunks WHERE path LIKE '%a.py'")]
    assert any("def g" in t for t in texts)
    conn.close()


def test_index_stores_logical_paths_not_local_ones(project):
    root, db_path = project
    index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)
    conn = store.connect(db_path)
    paths = {row[0] for row in conn.execute("SELECT path FROM chunks")} | {row[0] for row in conn.execute("SELECT path FROM files")}
    conn.close()
    assert paths == {"demo/a.py"}


def test_deleted_file_is_removed_from_index(project):
    root, db_path = project
    index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)

    a_py = root / "a.py"
    a_py.unlink()

    report4 = index_corpus([Repo("demo", root)], models=[FAKE_MODEL], db_path=db_path, verbose=False)
    assert report4.files_deleted == 1

    conn = store.connect(db_path)
    remaining = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert remaining == 0
    remaining_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert remaining_files == 0
    remaining_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert remaining_emb == 0
    conn.close()
