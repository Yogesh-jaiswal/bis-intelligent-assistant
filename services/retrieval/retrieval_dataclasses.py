from dataclasses import dataclass
from models.enums import DocumentTypes

from services.file_processors.document.doc_representation import DocumentBlock

@dataclass(slots=True)
class RetrievedChunk:
    """Represents a chunk of a document that has been retrieved, including its score and associated metadata."""
    score: float
    chunk: DocumentBlock

    filename: str
    author: str | None
    source_type: DocumentTypes
    source_url: str | None
    """Authoritative external URL from Upload.source_url. None when not available."""



@dataclass
class ContextBundle:
    """Represents a bundle of context information, including a list of retrieved chunks."""
    chunks: list[RetrievedChunk]

    def to_text(self) -> str:
        return "\n\n---\n\n".join(
            chunk.chunk.text
            for chunk in self.chunks
        )

@dataclass(frozen=True)
class Citation:
    """Represents a citation for a retrieved chunk, including the filename, source type, author, and associated metadata."""
    filename: str
    source_type: DocumentTypes
    author: str | None
    metadata: dict