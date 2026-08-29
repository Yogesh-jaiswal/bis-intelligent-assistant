from .retrieval_dataclasses import Citation, RetrievedChunk

class CitationBuilder:
    """Builds a list of unique citations from a list of retrieved chunks."""

    @staticmethod
    def build(
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:

        citations: list[Citation] = []

        seen: set = set()

        for retrieved in retrieved_chunks:

            citation = Citation(
                filename=retrieved.filename,
                author=retrieved.author,
                source_type=retrieved.source_type,
                metadata=retrieved.chunk.metadata,
            )

            key = (
                citation.filename,
                citation.source_type,
                tuple(sorted(citation.metadata.items()))
            )

            if key in seen:
                continue

            seen.add(key)
            citations.append(citation)

        citations.sort(
            key=CitationBuilder._sort_key
        )

        return citations

    @staticmethod
    def _sort_key(
        citation: Citation,
    ):

        metadata = citation.metadata

        return (
            metadata.get("page", float("inf")),
            metadata.get("start", float("inf")),
            metadata.get("row_range", "")
        )