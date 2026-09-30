"""Backend-selection logic in code_rag.embeddings — no real model is
loaded by these tests (no network, no CUDA, fast)."""
from code_rag import embeddings as emb
from code_rag import config


def test_plain_model_id_selects_fastembed_cpu_backend():
    base_id, backend = emb._parse_model("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    assert backend == "fastembed-cpu"
    assert base_id == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def test_st_cuda_suffix_selects_st_cuda_backend_and_strips_suffix():
    base_id, backend = emb._parse_model("intfloat/multilingual-e5-large@st-cuda")
    assert backend == "st-cuda"
    assert base_id == "intfloat/multilingual-e5-large"


def test_config_gpu_model_constant_round_trips_through_parse():
    base_id, backend = emb._parse_model(config.MODEL_E5_LARGE_GPU)
    assert backend == "st-cuda"
    assert base_id == config.MODEL_E5_LARGE


def test_e5_prefix_detection_is_backend_independent():
    # Same base id needs the "query: "/"passage: " e5 prefix regardless of
    # which backend serves it.
    cpu_base, _ = emb._parse_model(config.MODEL_E5_LARGE)
    gpu_base, _ = emb._parse_model(config.MODEL_E5_LARGE_GPU)
    assert emb._needs_e5_prefix(cpu_base)
    assert emb._needs_e5_prefix(gpu_base)
    assert cpu_base == gpu_base


def test_light_model_does_not_need_e5_prefix():
    base_id, _ = emb._parse_model(config.MODEL_BGE_SMALL)
    assert not emb._needs_e5_prefix(base_id)


def test_unload_model_is_safe_when_nothing_was_loaded():
    # Must not raise even though no real model was ever loaded/imported.
    emb.unload_model("some/model-that-was-never-loaded")
    emb.unload_model("some/model-that-was-never-loaded@st-cuda")
