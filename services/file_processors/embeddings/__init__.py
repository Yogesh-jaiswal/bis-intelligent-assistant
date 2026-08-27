"""
Document embedding services.

Provides interfaces and implementations for converting
document chunks into vector representations used by
retrieval and similarity-search components.
"""

from .embeddings_generator import EmbeddingGenerator
from .fake_embeddings import FakeEmbeddingGenerator

class EmbeddingFactory:
    """A factory class to create different types of embedding generators based on the specified type."""

    @staticmethod
    def get_provider(use_fake_emebedder: bool = False):
        return (
            FakeEmbeddingGenerator()
            if use_fake_emebedder
            else EmbeddingGenerator()
        )