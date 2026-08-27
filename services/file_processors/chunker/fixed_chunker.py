from .base_chunker import BaseChunker

class FixedSizeChunker(BaseChunker):
    """Divide the given text into given fixed size separate chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int= 100):
        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        chunks = []

        start = 0

        while (start < len(text)):
            end = start + self.chunk_size

            chunks.append(text[start:end])

            start += (
                self.chunk_size - self.overlap
            )

        return chunks