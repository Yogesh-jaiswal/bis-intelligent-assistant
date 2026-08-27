"""
Document chunking implementations.

Provides chunking strategies for dividing extracted document
content into smaller units suitable for retrieval and AI
processing.
"""

from typing import Literal
from .fixed_chunker import FixedSizeChunker
from .sentence_chunker import SentenceChunker
from .token_chunker import TokenChunker

class ChunkerFactory:
    """A factory class to create different types of chunkers based on the specified type."""

    def get_chunker(type: Literal["fixed", "sentence", "token"] = "fixed"):
        match type:
            case "fixed":
                return FixedSizeChunker()
            case "sentence":
                return SentenceChunker()
            case "token":
                return TokenChunker()
            case _:
                raise ValueError("Unknown chunker type")