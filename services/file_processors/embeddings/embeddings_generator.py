from sentence_transformers import SentenceTransformer

from configs import get_settings
from .base_embeddings_generator import BaseEmbeddingProvider, TokenEncoding

# Initialize the embedding model using the specified model from settings
MODEL = SentenceTransformer(
    get_settings().EMBEDDINGS_MODEL
)

class EmbeddingGenerator(BaseEmbeddingProvider):
    """Generates embeddings for text using a specified model and provides tokenization utilities."""

    def embed(self, text: str) -> list[float]:
        return MODEL.encode(text).tolist()
    
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return MODEL.encode(texts).tolist()

    def count_tokens(self, text: str) -> int:
        return len(
            MODEL.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    def offset_mapping(self, text: str) -> TokenEncoding:
        encoding = MODEL.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        return TokenEncoding(
            offset_mapping=encoding["offset_mapping"]
        )

    @property
    def max_tokens(self) -> int:
        return MODEL.max_seq_length