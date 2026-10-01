"""hybrid+rerank: the cross-encoder reorders the fused candidates. A fake scorer stands in for the
model, so the tests download nothing."""
import pytest

from code_rag import config, rerank, retrieval, store
from code_rag.chunker import Chunk


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "index.db")
    store.init_db(c)
    ids = store.insert_chunks(c, [
        Chunk("demo/tests/test_retry.py", "demo", "test_retry", "function", 1, 5, "def test_retry(): ..."),
        Chunk("demo/src/retry.py", "demo", "retry", "function", 1, 9, "def retry(fn): ..."),
        Chunk("demo/src/other.py", "demo", "other", "function", 1, 3, "def other(): ..."),
    ])
    c.commit()
    yield c, ids
    c.close()


def fake_scores(by_symbol: dict[str, float]):
    def score(query, passages, model_name=None):
        # Each passage starts with the "path | symbol | kind" header.
        return [next(v for k, v in by_symbol.items() if f"| {k} |" in p) for p in passages]
    return score


def test_candidates_are_reordered_by_the_cross_encoder(conn, monkeypatch):
    c, (test, impl, other) = conn
    monkeypatch.setattr(rerank, "score", fake_scores({"test_retry": 0.2, "retry": 0.9, "other": 0.5}))
    out = retrieval.rerank_candidates(c, "q", [(test, 3.0), (other, 2.0), (impl, 1.0)], k=10)
    assert [cid for cid, _ in out] == [impl, other, test]


def test_the_source_prior_applies_after_reranking(conn, monkeypatch):
    c, (test, impl, _) = conn
    monkeypatch.setattr(config, "NON_SOURCE_WEIGHT", 0.5)
    monkeypatch.setattr(rerank, "score", fake_scores({"test_retry": 0.8, "retry": 0.6, "other": 0.0}))
    out = retrieval.rerank_candidates(c, "q", [(test, 2.0), (impl, 1.0)], k=10)
    assert [cid for cid, _ in out] == [impl, test]  # 0.8 * 0.5 < 0.6


def test_candidates_past_the_window_keep_their_fused_order(conn, monkeypatch):
    c, (test, impl, other) = conn
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 1)
    monkeypatch.setattr(rerank, "score", fake_scores({"test_retry": 0.1, "retry": 0.9, "other": 0.9}))
    out = retrieval.rerank_candidates(c, "q", [(test, 3.0), (other, 2.0), (impl, 1.0)], k=10)
    assert [cid for cid, _ in out] == [test, other, impl]


def test_k_limits_the_reranked_list(conn, monkeypatch):
    c, (test, impl, other) = conn
    monkeypatch.setattr(rerank, "score", fake_scores({"test_retry": 0.1, "retry": 0.9, "other": 0.5}))
    assert [cid for cid, _ in retrieval.rerank_candidates(c, "q", [(test, 3.0), (other, 2.0), (impl, 1.0)], k=1)] == [impl]


def test_unknown_mode_is_rejected(conn):
    with pytest.raises(ValueError):
        retrieval.search(conn[0], "q", mode="nope")


def test_rerank_is_a_search_mode():
    assert "hybrid+rerank" in config.SEARCH_MODES


def test_unload_is_safe_when_nothing_was_loaded():
    rerank.unload()
    rerank.unload("some/reranker-that-was-never-loaded")
