from code_rag.retrieval import rrf_fuse


def test_rrf_combines_two_rankings_by_reciprocal_rank():
    # k=0 keeps the arithmetic exact: score = 1 / (rank + 1).
    list1 = [1, 2, 3]
    list2 = [3, 1, 2]
    fused = rrf_fuse([list1, list2], k=0)
    ids = [cid for cid, _ in fused]
    # 1: 1/1 + 1/2 = 1.5 ; 3: 1/3 + 1/1 = 1.333 ; 2: 1/2 + 1/3 = 0.833
    assert ids == [1, 3, 2]


def test_rrf_item_present_in_only_one_list_still_scored():
    fused = rrf_fuse([[10, 20], []], k=0)
    assert [cid for cid, _ in fused] == [10, 20]


def test_rrf_higher_rank_beats_lower_rank_in_single_list():
    fused = rrf_fuse([[5, 6, 7]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids == [5, 6, 7]


def test_rrf_empty_input():
    assert rrf_fuse([], k=60) == []


def test_rrf_agreement_across_lists_boosts_score_above_single_list_top():
    # An item ranked #2 in both lists should outscore an item ranked #1 in
    # only one list and absent from the other, once both lists agree.
    list1 = [100, 200]
    list2 = [300, 200]
    fused = dict(rrf_fuse([list1, list2], k=60))
    assert fused[200] > fused[100]
    assert fused[200] > fused[300]
