"""Per-file cap (config.MAX_CHUNKS_PER_FILE): one file's chunks must not crowd out every other file's
best chunk in the final top-k."""
from code_rag.retrieval import _cap_per_file


def test_cap_limits_chunks_per_file():
    # ids 1,2,3 all from fileA (best-ranked first), id 4 from fileB.
    scored = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    id_to_path = {1: "fileA", 2: "fileA", 3: "fileA", 4: "fileB"}
    out = _cap_per_file(scored, id_to_path, cap=2, k=10)
    ids = [cid for cid, _ in out]
    assert ids == [1, 2, 4]  # id 3 dropped: fileA already has 2


def test_cap_preserves_rank_order():
    scored = [(10, 5.0), (11, 4.0), (12, 3.0)]
    id_to_path = {10: "a", 11: "b", 12: "c"}
    out = _cap_per_file(scored, id_to_path, cap=2, k=10)
    assert [cid for cid, _ in out] == [10, 11, 12]


def test_cap_stops_once_k_results_collected():
    scored = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]
    id_to_path = {1: "a", 2: "b", 3: "c", 4: "d"}
    out = _cap_per_file(scored, id_to_path, cap=2, k=2)
    assert len(out) == 2
    assert [cid for cid, _ in out] == [1, 2]


def test_cap_of_one_keeps_only_best_chunk_per_file():
    scored = [(1, 0.9), (2, 0.8)]
    id_to_path = {1: "fileA", 2: "fileA"}
    out = _cap_per_file(scored, id_to_path, cap=1, k=10)
    assert [cid for cid, _ in out] == [1]


def test_cap_skips_ids_with_unknown_path():
    scored = [(1, 0.9), (2, 0.8)]
    id_to_path = {1: "fileA"}  # id 2 missing (e.g. deleted mid-query)
    out = _cap_per_file(scored, id_to_path, cap=2, k=10)
    assert [cid for cid, _ in out] == [1]
