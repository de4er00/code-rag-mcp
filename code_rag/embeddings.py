"""Local embedding backends: no API keys, no network calls beyond downloading model weights once.

- fastembed (onnxruntime, CPU) is the default and needs no GPU.
- sentence-transformers on CUDA in fp16 is selected by the "@st-cuda" suffix, e.g.
  "intfloat/multilingual-e5-large@st-cuda". The suffix makes it a different model id, so GPU and
  CPU vectors of the same base model never mix in storage, and a query is always embedded by the
  same backend as the passages it is compared with.

Models are unloaded explicitly after indexing: two transformer models plus onnxruntime buffers do
not fit comfortably in 16 GB of RAM.
"""
from __future__ import annotations

import gc

import numpy as np

from . import config

_MODEL_CACHE: dict = {}
_ST_MODEL_CACHE: dict = {}

ST_CUDA_SUFFIX = "@st-cuda"

# Conservative per-forward-pass batch sizes for onnxruntime on CPU. Bigger
# models get a smaller batch to keep peak RSS down; fastembed batches
# internally when `batch_size` is passed to `.embed()`, so the caller-side
# batch (see indexer.py) can still be larger — this only controls how many
# texts go through one ONNX forward pass at a time.
ONNX_BATCH_SIZE = {
    config.MODEL_E5_LARGE: 16,
    config.MODEL_BGE_SMALL: 32,
}
DEFAULT_ONNX_BATCH_SIZE = 16

ST_CUDA_BATCH_SIZE = 32


def _parse_model(model_name: str) -> tuple[str, str]:
    """Returns (base_model_id, backend), backend in {'fastembed-cpu', 'st-cuda'}."""
    if model_name.endswith(ST_CUDA_SUFFIX):
        return model_name[: -len(ST_CUDA_SUFFIX)], "st-cuda"
    return model_name, "fastembed-cpu"


def _needs_e5_prefix(base_model_id: str) -> bool:
    return "e5" in base_model_id.lower()


def _cache_dir() -> str:
    d = config.PROJECT_ROOT / "model_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def get_model(base_model_id: str):
    """fastembed (CPU/onnxruntime) model, keyed by base model id."""
    if base_model_id not in _MODEL_CACHE:
        from fastembed import TextEmbedding

        _MODEL_CACHE[base_model_id] = TextEmbedding(model_name=base_model_id, cache_dir=_cache_dir())
    return _MODEL_CACHE[base_model_id]


def get_st_cuda_model(base_model_id: str):
    """sentence-transformers model on CUDA, fp16, keyed by base model id."""
    if base_model_id not in _ST_MODEL_CACHE:
        import torch
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the st-cuda embedding backend")
        _ST_MODEL_CACHE[base_model_id] = SentenceTransformer(
            base_model_id,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
            cache_folder=_cache_dir(),
        )
    return _ST_MODEL_CACHE[base_model_id]


def unload_model(model_name: str) -> None:
    """Drop a loaded model (either backend) from cache and free its memory.
    Safe to call even if the model was never loaded."""
    base_id, backend = _parse_model(model_name)
    if backend == "st-cuda":
        _ST_MODEL_CACHE.pop(base_id, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        _MODEL_CACHE.pop(base_id, None)
        gc.collect()


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_passages(model_name: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    base_id, backend = _parse_model(model_name)
    prefixed = [f"passage: {t}" for t in texts] if _needs_e5_prefix(base_id) else list(texts)

    if backend == "st-cuda":
        model = get_st_cuda_model(base_id)
        vecs = model.encode(
            prefixed, batch_size=ST_CUDA_BATCH_SIZE, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
    else:
        model = get_model(base_id)
        onnx_batch = ONNX_BATCH_SIZE.get(base_id, DEFAULT_ONNX_BATCH_SIZE)
        vecs = np.array(list(model.embed(prefixed, batch_size=onnx_batch)), dtype=np.float32)

    return _normalize(vecs)


def embed_query(model_name: str, text: str) -> np.ndarray:
    base_id, backend = _parse_model(model_name)
    prefixed = f"query: {text}" if _needs_e5_prefix(base_id) else text

    if backend == "st-cuda":
        model = get_st_cuda_model(base_id)
        vec = model.encode([prefixed], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        vec = np.asarray(vec, dtype=np.float32)
    else:
        model = get_model(base_id)
        vec = np.array(list(model.embed([prefixed]))[0], dtype=np.float32)

    return _normalize(vec.reshape(1, -1))[0]
