from .base_embeddings_generator import BaseEmbeddingProvider, TokenEncoding


class FakeEmbeddingGenerator(BaseEmbeddingProvider):
    """A mock embedding generator that provides fake embeddings and tokenization for testing purposes."""

    def embed(self, text: str) -> list[float]:
        return [0.01] * 384

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 384 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return len(text)

    def offset_mapping(self, text: str) -> TokenEncoding:
        return TokenEncoding(
            offset_mapping=[
                (i, i + 1)
                for i in range(len(text))
            ]
        )

    def tokenize(self, text: str) -> TokenEncoding:
        return self.offset_mapping(text)

    @property
    def max_tokens(self) -> int:
        return 1000