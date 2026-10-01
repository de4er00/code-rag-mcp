"""Optional cross-encoder reranker for the "hybrid+rerank" mode.

A cross-encoder reads the question and a candidate chunk together, so it can tell the function
that implements something from a chunk that only mentions it. It is slower than the retrievers,
so it only reorders the top fused candidates.

- fastembed (onnxruntime, CPU) is the default backend, like for embeddings.
- The "@st-cuda" suffix selects sentence-transformers on a CUDA GPU in fp16.

The model is loaded on the first reranked query and never by the other modes, so installing and
testing the package does not download it.
"""
from __future__ import annotations

import gc
import math

from . import config
from .embeddings import ST_CUDA_SUFFIX, _cache_dir

_MODELS: dict = {}


def _load(model_name: str):
    if model_name.endswith(ST_CUDA_SUFFIX):
        import torch
        from sentence_transformers import CrossEncoder

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the st-cuda reranker backend")
        return CrossEncoder(
            model_name[: -len(ST_CUDA_SUFFIX)],
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
            cache_folder=_cache_dir(),
            max_length=512,
        )
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=_cache_dir())


def score(query: str, passages: list[str], model_name: str | None = None) -> list[float]:
    """Relevance of each passage to the query in (0, 1), higher is better."""
    if not passages:
        return []
    name = model_name or config.RERANK_MODEL
    if name not in _MODELS:
        _MODELS[name] = _load(name)
    model = _MODELS[name]
    if name.endswith(ST_CUDA_SUFFIX):
        # sentence-transformers already applies a sigmoid to single-label models.
        out = model.predict([(query, p) for p in passages], batch_size=config.RERANK_BATCH_SIZE,
                            convert_to_numpy=True, show_progress_bar=False)
        return [float(s) for s in out]
    # fastembed returns logits; the sigmoid keeps scores positive so the source prior can scale them.
    logits = model.rerank(query, passages, batch_size=config.RERANK_BATCH_SIZE)
    return [0.5 * (1.0 + math.tanh(float(x) / 2)) for x in logits]


def unload(model_name: str | None = None) -> None:
    """Drop a loaded reranker (or all of them) and free its memory."""
    if model_name is None:
        _MODELS.clear()
    else:
        _MODELS.pop(model_name, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
