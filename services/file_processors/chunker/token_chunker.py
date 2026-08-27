from .base_chunker import BaseChunker

from services.file_processors.document.doc_representation import DocumentBlock

class TokenChunker(BaseChunker):
    """Divide the given text into token based separate chunks."""

    def __init__(self, embeddings_provider):
        self.provider = embeddings_provider

        self.max_tokens = embeddings_provider.max_tokens // 2
        self.overlap_tokens = self.max_tokens // 3

    def chunk_text(self, block: DocumentBlock) -> list[DocumentBlock]:
        text = block.text

        offsets = self.provider.offset_mapping(text).offset_mapping

        chunks = []

        token_start = 0

        while token_start < len(offsets):

            token_end = min(
                token_start + self.max_tokens,
                len(offsets),
            )

            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]

            chunks.append(
                block.copy(
                    text=text[char_start: char_end],
                    metadata={
                        "token_range": {
                            "start": token_start,
                            "end": token_end
                        }
                    }
                )
            )

            if token_end == len(offsets):
                break

            token_start += (
                self.max_tokens
                - self.overlap_tokens
            )

        return chunks