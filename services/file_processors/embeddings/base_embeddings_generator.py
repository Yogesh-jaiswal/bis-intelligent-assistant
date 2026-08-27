from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TokenEncoding:
    """Represents the tokenization of a text, including the tokens and their corresponding offset mappings."""
    offset_mapping: list[tuple[int, int]]

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers, defining the interface for generating embeddings and tokenization."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the total number of tokens in the given text block."""
        pass

    @abstractmethod
    def offset_mapping(self, text: str) -> TokenEncoding:
        """Return the offset mapping of the using the tokenizer associated with this embedding model."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum supported input tokens for this embedding model."""