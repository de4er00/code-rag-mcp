"""hybrid+rerank: the cross-encoder's order is fused with the retrieval order. A fake scorer stands
in for the model, so the tests download nothing."""
import pytest

from code_rag import config, rerank, retrieval, store
from code_rag.chunker import Chunk

NAMES = ("test_retry", "retry", "other", "alpha", "beta")


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "index.db")
    store.init_db(c)
    chunks = [Chunk("demo/tests/test_retry.py", "demo", "test_retry", "function", 1, 5, "def test_retry(): ...")]
    chunks += [Chunk(f"demo/src/{n}.py", "demo", n, "function", 1, 5, f"def {n}(): ...") for n in NAMES[1:]]
    ids = dict(zip(NAMES, store.insert_chunks(c, chunks)))
    c.commit()
    yield c, ids
    c.close()


def fake_scores(by_symbol: dict[str, float]):
    def score(query, passages, model_name=None):
        # Each passage starts with the "path | symbol | kind" header.
        return [next(v for k, v in by_symbol.items() if f"| {k} |" in p) for p in passages]
    return score


def order(conn, monkeypatch, names, scores, k=10):
    c, ids = conn
    monkeypatch.setattr(rerank, "score", fake_scores(scores))
    out = retrieval.rerank_candidates(c, "q", [(ids[n], 1.0) for n in names], k=k)
    by_id = {v: n for n, v in ids.items()}
    return [by_id[cid] for cid, _ in out]


def test_a_candidate_the_cross_encoder_likes_moves_up_but_not_alone(conn, monkeypatch):
    # Fourth in retrieval order, first for the cross-encoder: fused, it lands second.
    scores = {"alpha": 0.6, "beta": 0.5, "other": 0.4, "retry": 0.9, "test_retry": 0.0}
    names = ["alpha", "beta", "other", "retry", "test_retry"]
    assert order(conn, monkeypatch, names, scores) == ["alpha", "retry", "beta", "other", "test_retry"]


def test_ties_go_to_the_cross_encoder(conn, monkeypatch):
    # Retrieval and cross-encoder disagree symmetrically, so both ends tie under RRF.
    scores = {"test_retry": 0.2, "retry": 0.9, "other": 0.5}
    assert order(conn, monkeypatch, ["test_retry", "other", "retry"], scores) == ["retry", "test_retry", "other"]


def test_the_source_prior_applies_to_the_cross_encoder_scores(conn, monkeypatch):
    monkeypatch.setattr(config, "NON_SOURCE_WEIGHT", 0.5)
    scores = {"test_retry": 0.8, "retry": 0.6}
    assert order(conn, monkeypatch, ["test_retry", "retry"], scores) == ["retry", "test_retry"]  # 0.8 * 0.5 < 0.6


def test_candidates_past_the_window_keep_their_order(conn, monkeypatch):
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 1)
    scores = {"test_retry": 0.1, "retry": 0.9, "other": 0.9}
    assert order(conn, monkeypatch, ["test_retry", "other", "retry"], scores) == ["test_retry", "other", "retry"]


def test_k_limits_the_reranked_list(conn, monkeypatch):
    scores = {"test_retry": 0.1, "retry": 0.9, "other": 0.5}
    assert order(conn, monkeypatch, ["test_retry", "other", "retry"], scores, k=1) == ["retry"]


def test_unknown_mode_is_rejected(conn):
    with pytest.raises(ValueError):
        retrieval.search(conn[0], "q", mode="nope")


def test_rerank_is_a_search_mode():
    assert "hybrid+rerank" in config.SEARCH_MODES


def test_unload_is_safe_when_nothing_was_loaded():
    rerank.unload()
    rerank.unload("some/reranker-that-was-never-loaded")
