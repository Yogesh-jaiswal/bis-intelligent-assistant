from services.file_processors.chunker.token_chunker import TokenChunker

from .doc_representation import (
    DocumentRepresentation,
    DocumentBlock,
)

from models.enums import DocumentBlockType

# Blocks that should not be split into smaller chunks, regardless of their token count
NON_SPLITTABLE_BLOCKS = {
    DocumentBlockType.TABLE,
    DocumentBlockType.HEADING,
}


class DocumentChunker:
    """Splits document blocks into embedding-ready chunks while preserving metadata."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.token_chunker = TokenChunker(embedder)

    def split(
        self,
        doc_repr: DocumentRepresentation,
    ) -> list[DocumentBlock]:

        chunks = []

        for block in doc_repr.blocks:

            token_count = self.embedder.count_tokens(block.text)

            # Block fits inside one chunk
            if (
                block.type in NON_SPLITTABLE_BLOCKS
                or token_count <= self.token_chunker.max_tokens
            ):
                chunks.append(block)
                continue

            # Split oversized block
            chunks.extend(self.token_chunker.chunk_text(block))

        return chunks