"""
Sentence chunker — prototype implementation.

Uses the existing FixedSizeChunker internally so the factory can
instantiate it without requiring NLTK or spaCy.  A proper sentence-boundary
chunker can replace this implementation later without changing the interface.
"""

from .fixed_chunker import FixedSizeChunker
from .base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Prototype sentence chunker.

    Falls back to fixed-size splitting so the rest of the pipeline
    can run without NLP dependencies.  Replace the body of ``chunk_text``
    with real sentence-boundary splitting when NLTK / spaCy is available.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self._inner = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk_text(self, text: str) -> list[str]:
        return self._inner.chunk_text(text)
