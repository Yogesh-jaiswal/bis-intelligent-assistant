from pathlib import Path
from abc import ABC, abstractmethod

from services.file_processors.document.doc_representation import DocumentRepresentation

class BaseProcessor(ABC):
    
    @abstractmethod
    def extract(self, file_path: str | Path) -> DocumentRepresentation:
        """Extract text and available metadata from the file using given file path and creates a document representation"""
        pass