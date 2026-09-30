"""Source prior: tests, docs and changelogs rank below implementation files, without being dropped."""
import pytest

from code_rag import config
from code_rag.retrieval import _apply_prior, rrf_fuse, source_prior


@pytest.mark.parametrize("path", [
    "httpx/tests/client/test_redirects.py",
    "requests/docs/user/quickstart.rst",
    "rich/CHANGELOG.md",
    "rich/questions/square_brackets.question.md",
    "pkg/module_test.py",
])
def test_non_source_paths_are_down_weighted(path):
    assert source_prior(path) == config.NON_SOURCE_WEIGHT < 1.0


@pytest.mark.parametrize("path", [
    "click/src/click/testing.py",
    "requests/src/requests/sessions.py",
    "rich/rich/markup.py",
])
def test_source_paths_keep_full_weight(path):
    assert source_prior(path) == 1.0


def test_prior_reorders_but_keeps_everything():
    paths = {1: "a/tests/test_x.py", 2: "a/src/x.py"}
    reranked = _apply_prior([(1, 1.0), (2, 0.8)], paths)
    assert [cid for cid, _ in reranked] == [2, 1]


def test_weighted_rrf_favours_the_heavier_list():
    fused = rrf_fuse([[1, 2], [2, 1]], k=60, weights=[0.5, 1.0])
    assert fused[0][0] == 2
