"""
CitationBuilder — constructs citation objects from retrieved document evidence.

Two public methods are provided:

* ``build()``  — internal deduplication (used for ordering / inspection).
* ``build_api_citations()``  — produces the API ``validators.chat_responses.Citation``
  Pydantic objects suitable for inclusion in a ``ChatResponse``.

URL fabrication rule
--------------------
``build_api_citations`` will ONLY include a chunk in the citation list when it can
resolve a real ``source_url``.  It looks up (in order):

1. ``chunk.source_url``          — from ``Upload.source_url``
2. manifest lookup by ``document_id`` in chunk metadata
3. (no fallback — chunk is excluded)

The generic BIS homepage is NOT used as a fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .retrieval_dataclasses import Citation, RetrievedChunk

if TYPE_CHECKING:
    from validators.chat_responses import Citation as ApiCitation
    from services.ingestion.manifest import ManifestEntry


class CitationBuilder:
    """Builds citation objects from a list of retrieved chunks."""

    # ------------------------------------------------------------------
    # Original internal build (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def build(
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """
        Return deduplicated internal ``Citation`` dataclass objects.

        These are used for internal ordering and inspection only —
        they are NOT the API response model.
        """
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
                tuple(sorted(citation.metadata.items())),
            )

            if key in seen:
                continue

            seen.add(key)
            citations.append(citation)

        citations.sort(key=CitationBuilder._sort_key)
        return citations

    @staticmethod
    def _sort_key(citation: Citation):
        return citation.metadata.get("page", float("inf"))

    # ------------------------------------------------------------------
    # API citation builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_api_citations(
        retrieved_chunks: list[RetrievedChunk],
        manifest_entries: list["ManifestEntry"] | None = None,
    ) -> list["ApiCitation"]:
        """
        Build ``validators.chat_responses.Citation`` Pydantic objects from
        retrieved chunks.

        Deduplication key: **filename** (not the full metadata dict), so all
        chunks from the same document map to a single citation.  When multiple
        pages are present the reference lists the first page seen.

        Chunks for which no real ``source_url`` can be resolved are silently
        excluded — no fake URLs are ever produced.

        :param retrieved_chunks: RAG results from ``SimilaritySearchService``.
        :param manifest_entries: Optional manifest for URL lookup by document_id.
        :return: List of ``validators.chat_responses.Citation`` objects.
        """
        # Lazy import to avoid circular dependencies
        from validators.chat_responses import Citation as ApiCitation  # noqa: PLC0415

        # Build manifest lookup: document_id → source_url
        manifest_url_map: dict[str, str] = {}
        if manifest_entries:
            for entry in manifest_entries:
                if entry.source_url:
                    manifest_url_map[entry.document_id] = entry.source_url

        # Per-filename aggregation: first page and resolved URL
        seen_filenames: dict[str, dict] = {}  # filename → aggregated data

        for chunk in retrieved_chunks:
            fname = chunk.filename
            if fname in seen_filenames:
                continue  # already have a citation for this document

            # Resolve URL: chunk.source_url → manifest → DB Standard lookup → skip
            resolved_url: str | None = chunk.source_url
            if not resolved_url:
                doc_id = chunk.chunk.metadata.get("document_id")
                if doc_id and doc_id in manifest_url_map:
                    resolved_url = manifest_url_map[doc_id]

            if not resolved_url:
                # Check chunk metadata for explicit standard number or URL
                is_num = chunk.chunk.metadata.get("is_number") or chunk.chunk.metadata.get("standard_number")
                if not is_num and fname:
                    from services.ingestion.manifest import _parse_pdf_filename
                    is_num, _ = _parse_pdf_filename(fname)

                if is_num:
                    try:
                        from app.extensions import db
                        from models.standard import Standard
                        from sqlalchemy import select
                        clean_num = is_num.replace("IS", "").strip()
                        std_rec = db.session.execute(
                            select(Standard).where(Standard.is_number.ilike(f"%{clean_num}%"))
                        ).scalars().first()
                        if std_rec and (std_rec.document_url or std_rec.source_url):
                            resolved_url = std_rec.document_url or std_rec.source_url
                    except Exception:
                        pass

            if not resolved_url:
                # Cannot produce a valid Citation.source_url — skip this chunk.
                continue

            page: int | None = chunk.chunk.metadata.get("page")
            seen_filenames[fname] = {
                "source_url": resolved_url,
                "page": page,
                "author": chunk.author,
            }

        # Build output list in stable order (order of first appearance)
        api_citations: list[ApiCitation] = []
        cit_index = 1

        for fname, data in seen_filenames.items():
            source_url: str = data["source_url"]
            page: int | None = data["page"]

            # Build a human-readable title from the filename
            stem = fname.rsplit(".", 1)[0] if "." in fname else fname
            title = stem.replace("_", " ").strip()

            # Build reference string
            if page is not None:
                reference = f"{title}, page {page}"
            else:
                reference = title

            try:
                citation = ApiCitation(
                    id=f"cit_{cit_index}",
                    source_type="document",
                    title=title,
                    reference=reference,
                    source_url=source_url,  # type: ignore[arg-type]
                )
                api_citations.append(citation)
                cit_index += 1
            except Exception:
                # Pydantic validation failed (e.g. source_url is not a valid URL)
                # Skip rather than crash.
                continue

        return api_citations