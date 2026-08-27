from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

from models.enums import DocumentBlockType

# Type alias for metadata associated with a document block
DocumentBlockMetadata = dict[str, Any]

@dataclass
class DocumentBlock:
    """Represents a block of text in a document, along with its type and associated metadata."""
    type: DocumentBlockType
    text: str
    metadata: dict = field(default_factory=dict)

    def copy(self, **updates):
        return DocumentBlock(
            type=updates.get("type", self.type),
            text=updates.get("text", self.text),
            metadata={
                **deepcopy(self.metadata),
                **updates.get("metadata", {}),
            }
        )

@dataclass
class DocumentRepresentation:
    """Represents a document as a collection of blocks, along with the author's information."""
    blocks: list[DocumentBlock]
    author: str | None = field(default=None)

    def to_text(self):
        return "\n\n".join(
            block.text
            for block in self.blocks
        )