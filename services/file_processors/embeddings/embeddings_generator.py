import logging
import time
import torch
from sentence_transformers import SentenceTransformer

from configs import get_settings
from .base_embeddings_generator import BaseEmbeddingProvider, TokenEncoding

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None
_DEVICE: str | None = None


def get_embedding_device() -> str:
    """Determine the device to use for embeddings (CUDA if available, else CPU)."""
    settings = get_settings()
    configured_device = getattr(settings, "EMBEDDINGS_DEVICE", None)
    if configured_device:
        return configured_device.lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_embedding_model(force_reload: bool = False) -> SentenceTransformer | None:
    """
    Explicitly load and initialize the SentenceTransformer embedding model at application startup.
    Reuses the already loaded model singleton across requests to eliminate query-time latency spikes.
    """
    global _MODEL, _DEVICE
    if _MODEL is not None and not force_reload:
        return _MODEL

    settings = get_settings()
    model_name = settings.EMBEDDINGS_MODEL
    _DEVICE = get_embedding_device()

    logger.info("[EMBEDDING MODEL] Using device='%s'", _DEVICE)
    logger.info("[EMBEDDING MODEL] Loading model '%s'...", model_name)
    start_t = time.perf_counter()

    try:
        _MODEL = SentenceTransformer(model_name, device=_DEVICE)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        logger.info("[EMBEDDING MODEL] Loaded successfully in %.2f ms", elapsed_ms)
        return _MODEL
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        logger.error(
            "[EMBEDDING MODEL] Failed to load model '%s' after %.2f ms: %s",
            model_name,
            elapsed_ms,
            exc,
        )
        raise


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = init_embedding_model()
    return _MODEL


class EmbeddingGenerator(BaseEmbeddingProvider):
    """Generates embeddings for text using a pre-initialized SentenceTransformer model."""

    def embed(self, text: str) -> list[float]:
        return _get_model().encode(text).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return _get_model().encode(texts).tolist()

    def count_tokens(self, text: str) -> int:
        return len(
            _get_model().tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def offset_mapping(self, text: str) -> TokenEncoding:
        encoding = _get_model().tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        return TokenEncoding(
            offset_mapping=encoding["offset_mapping"]
        )

    @property
    def max_tokens(self) -> int:
        return _get_model().max_seq_length