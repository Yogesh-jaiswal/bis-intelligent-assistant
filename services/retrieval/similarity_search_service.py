from repositories.embedding_repository import retrieve_similar_chunks
from services.file_processors.embeddings.embeddings_generator import EmbeddingGenerator

from services.file_processors.document.doc_representation import DocumentBlock

from .retrieval_dataclasses import RetrievedChunk

class SimilaritySearchService:
    """Service for performing similarity searches on document chunks based on embeddings."""
    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def search(self, query: str, k: int = 5, upload_id: str | None = None) -> list[RetrievedChunk]:
        """
        Search query in stored embeddings and return top k results
        """
        query_embedding = self.embedder.embed(query)

        result = retrieve_similar_chunks(query_embedding, k=k, upload_id=upload_id)

        return [
            RetrievedChunk(
                score=score,
                chunk=DocumentBlock(
                    type=chunk.block_type,
                    text=chunk.content,
                    metadata=chunk.chunk_metadata
                ),
                filename=upload.filename,
                author=upload.author,
                source_type=upload.source_type
            ) for (score, chunk, upload) in result
        ]