from abc import ABC, abstractmethod

class BaseChunker(ABC):

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """Returns list of chunks according to the chosen chunking technique"""
        pass