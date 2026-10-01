"""Paths, file selection rules, model names and retrieval constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "index.db"
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "corpus.yaml"


@dataclass(frozen=True)
class Repo:
    name: str
    root: Path


def load_corpus(path: Path = DEFAULT_CORPUS_FILE) -> list[Repo]:
    """Read the list of repositories to index; relative paths resolve against the YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = Path(path).resolve().parent
    return [Repo(r["name"], (base / r["path"]).resolve()) for r in data.get("repos", [])]


INCLUDE_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".j2"}
MAX_FILE_SIZE = 300 * 1024  # bytes

SKIP_DIR_NAMES = {
    ".venv", "venv", "site-packages", "node_modules", ".git", "__pycache__", "dist", "build",
    ".pytest_cache", ".cache", ".idea", ".mypy_cache", ".ruff_cache", ".tox", ".nox", "htmlcov",
}
# Any directory whose relative path contains one of these is treated as vendored or generated.
SKIP_DIR_PATH_SUBSTRINGS = ["vendor", "third_party", "thirdparty", "_generated"]

# Files that look like they hold credentials are never read, whatever their extension.
SKIP_FILE_SUBSTRINGS = ["credential", "secret", ".env"]
SKIP_FILE_SUFFIXES = [".pem", ".key", ".p12"]

# Embedding models. The default runs on CPU through fastembed (ONNX). The "@st-cuda" suffix selects
# sentence-transformers on a CUDA GPU in fp16; it is a separate model id so its vectors never mix
# with CPU vectors of the same base model.
MODEL_BGE_SMALL = "BAAI/bge-small-en-v1.5"
MODEL_E5_LARGE = "intfloat/multilingual-e5-large"
MODEL_E5_LARGE_GPU = MODEL_E5_LARGE + "@st-cuda"
DEFAULT_MODEL = MODEL_BGE_SMALL

RRF_K = 60
# Weight of the keyword list in hybrid fusion, relative to the dense list.
BM25_RRF_WEIGHT = 0.5
# Score multiplier for tests, docs and changelogs (1.0 turns the source prior off).
NON_SOURCE_WEIGHT = 0.5

# Optional cross-encoder reranker for the "hybrid+rerank" mode, over the top fused candidates.
# The default runs on CPU through fastembed (ONNX) and is loaded only when that mode is used.
# The "@st-cuda" suffix selects sentence-transformers on a CUDA GPU in fp16, as for embeddings.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_GPU = "BAAI/bge-reranker-v2-m3@st-cuda"
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 8
SEARCH_MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")

# An exact filename or symbol match (header column) is a much stronger signal than one stemmed
# word in the body.
HEADER_BM25_WEIGHT = 3.0
BODY_BM25_WEIGHT = 1.0

# At most this many chunks from one file in a result list, so one file's tests and docs cannot
# crowd out every other file's best chunk.
MAX_CHUNKS_PER_FILE = 2
RETRIEVAL_POOL_MULTIPLIER = 6
RETRIEVAL_MIN_POOL = 30
