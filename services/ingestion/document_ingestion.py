"""
services/ingestion/document_ingestion.py
=========================================

Orchestrates the full pipeline for a single document source:

    Upload record (PDF or HTML)
        ↓
    Deduplication check (skip if chunks already exist)
        ↓
    FileProcessor  (extract → chunk → embed)
        ↓
    DocumentChunk + ChunkEmbedding rows committed to DB

Used by ``seed_dataset`` for both local PDFs and web pages.

Deduplication
-------------
Before processing any source, we check whether an ``Upload`` record for
that file/source_id already has ``DocumentChunk`` rows.  If it does, the
source is skipped entirely — no duplicate chunks or embeddings are created.

This means running ``flask seed_dataset`` multiple times is safe.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import select

from app.extensions import db
from models.document_chunk import DocumentChunk
from models.upload import Upload
from services.file_processors import FileProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_document(
    *,
    file_path: str | Path,
    file_type: str,
    filename: str,
    source_url: str | None = None,
    fake_embedder: bool = False,
) -> tuple[str, int]:
    """
    Ingest a single document (PDF or HTML) into the vector store.

    Idempotent: if an ``Upload`` for *filename* already exists and already
    has ``DocumentChunk`` rows, the call returns immediately with status
    ``"skipped"``.

    :param file_path: Absolute path to the file on disk.
    :param file_type: ``"pdf"`` or ``"html"``.
    :param filename: Human-readable name stored on the Upload record.
    :param source_url: Optional authoritative URL; stored on the Upload record.
    :param fake_embedder: When True, uses the zero-cost fake embedder (tests/dev).
    :returns: Tuple of (status, chunk_count) where status is
              ``"created"``, ``"skipped"``, or ``"failed"``.
    """
    file_path = Path(file_path)
    logger.info("[INGESTION: START] Processing document '%s' (type='%s', size=%s bytes)", filename, file_type, file_path.stat().st_size if file_path.exists() else "N/A")

    # ------------------------------------------------------------------
    # 1. Find-or-create Upload record
    # ------------------------------------------------------------------
    upload = db.session.execute(
        select(Upload).where(Upload.filename == filename)
    ).scalar_one_or_none()

    if upload is None:
        upload = Upload(
            id=str(uuid.uuid4()),
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            source_url=source_url,
        )
        db.session.add(upload)
        db.session.flush()
        logger.info("[INGESTION: DB] Created Upload record %s for '%s'", upload.id, filename)
    else:
        if source_url and not upload.source_url:
            upload.source_url = source_url
            logger.info("[INGESTION: DB] Updated source_url for existing Upload %s", upload.id)

    # ------------------------------------------------------------------
    # 2. Deduplication — skip if chunks already exist
    # ------------------------------------------------------------------
    existing_chunk_count = db.session.execute(
        select(DocumentChunk).where(DocumentChunk.upload_id == upload.id).limit(1)
    ).scalar_one_or_none()

    if existing_chunk_count is not None:
        logger.info("[INGESTION: DEDUP] Upload '%s' already has indexed chunks -> skipping re-processing", filename)
        db.session.commit()
        return "skipped", 0

    # ------------------------------------------------------------------
    # 3. Process: extract → chunk → embed
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    try:
        logger.info("[INGESTION: PROCESSOR] Extracting, chunking and embedding '%s'...", filename)
        processor = FileProcessor(file_type=file_type, fake_embedder=fake_embedder)
        processed = processor.process(str(file_path))
        elapsed_proc_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info("[INGESTION: PROCESSOR] Extracted %d chunks in %.2f ms for '%s'", len(processed.chunks), elapsed_proc_ms, filename)
    except Exception as exc:
        logger.error("[INGESTION: PROCESSOR ERROR] Failed to process '%s': %s", filename, exc, exc_info=True)
        db.session.rollback()
        return "failed", 0

    # ------------------------------------------------------------------
    # 4. Store DocumentChunk + ChunkEmbedding rows
    # ------------------------------------------------------------------
    try:
        from models.chunk_embeddings import ChunkEmbedding  # noqa: PLC0415

        for idx, proc_chunk in enumerate(processed.chunks):
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                upload_id=upload.id,
                chunk_index=idx,
                block_type=proc_chunk.block.type,
                content=proc_chunk.text,
                chunk_metadata=proc_chunk.metadata,
            )
            db.session.add(chunk)
            db.session.flush()

            embedding = ChunkEmbedding(
                id=str(uuid.uuid4()),
                chunk_id=chunk.id,
                vector=proc_chunk.embedding,
            )
            db.session.add(embedding)

        db.session.commit()
        logger.info(
            "[INGESTION: SUCCESS] Successfully committed %d chunks + vector embeddings for '%s'",
            len(processed.chunks),
            filename,
        )
        return "created", len(processed.chunks)

    except Exception as exc:
        logger.error("[INGESTION: DB ERROR] Failed to store chunks/embeddings for '%s': %s", filename, exc, exc_info=True)
        db.session.rollback()
        return "failed", 0
