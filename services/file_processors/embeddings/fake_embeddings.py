from .base_embeddings_generator import BaseEmbeddingProvider, TokenEncoding

class FakeEmbeddingGenerator(BaseEmbeddingProvider):
    """A mock embedding generator that provides fake embeddings and tokenization for testing purposes."""

    def embed(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_many(self, texts: list[str]):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def count_tokens(self, text):
        return len(text)

    def tokenize(self, text: str) -> TokenEncoding:
        return TokenEncoding(
            offset_mapping=[
                (i, i + 1)
                for i in range(len(text))
            ]
        )

    @property
    def max_tokens(self):
        return 1000