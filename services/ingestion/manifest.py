"""
Document manifest for the BIS Intelligent Assistant.

Tracks every known source document (local PDF, scraped page, or unavailable)
with a stable identifier so the rest of the pipeline can resolve citations
without fabricating URLs.

Manifest location: data/documents/manifest.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SourceStatus = Literal[
    "available",        # local file exists and is readable
    "downloaded",       # fetched from a URL, stored locally
    "scraped",          # web page saved to data/web/
    "unavailable",      # source known but cannot be obtained
    "failed",           # download/fetch attempt failed
    "requires_auth",    # page requires login or CAPTCHA
]

SourceType = Literal[
    "standard",         # BIS / IS standard document
    "certification",    # certification-related document
    "web_page",         # scraped public web page
    "document",         # general uploaded document
    "other",
]


@dataclass
class ManifestEntry:
    """One row in the document manifest."""

    document_id: str
    """Stable identifier, e.g. 'bis_is_694_2010'."""

    filename: str | None = None
    """Local filename, e.g. '694_2010_reff2020.pdf'."""

    standard_number: str | None = None
    """IS number when applicable, e.g. 'IS 694'."""

    revision: str | None = None
    """Publication year or revision string, e.g. '2010'."""

    source_type: SourceType = "document"

    source_url: str | None = None
    """Authoritative external URL. None when not known or not verified."""

    local_path: str | None = None
    """Relative path to the local file, e.g. 'BIS_documents/694_2010_reff2020.pdf'."""

    status: SourceStatus = "available"

    downloaded: bool = False
    """True when the file was obtained by fetching source_url."""

    title: str | None = None
    """Human-readable title when known."""

    notes: str | None = None
    """Free-text notes, e.g. reason for failure."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """
    Load the manifest from *path*.

    Returns an empty list (and leaves the file untouched) if the file is
    missing or empty.
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw or raw == "[]":
        return []
    data = json.loads(raw)
    return [ManifestEntry.from_dict(d) for d in data]



def save_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    """Serialise and write the manifest atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        [e.to_dict() for e in entries],
        indent=2,
        ensure_ascii=False,
    )
    p.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_entry(entries: list[ManifestEntry], new_entry: ManifestEntry) -> tuple[list[ManifestEntry], bool]:
    """
    Insert *new_entry* if its document_id is not present; otherwise update it.

    Update semantics (non-destructive):
    - A non-null field in *new_entry* **always** overwrites the stored value.
    - A **null** field in *new_entry* preserves the existing non-null value.
    - This prevents re-discovery from clearing metadata (e.g. a ``source_url``)
      that was set by a previous richer ingestion pass.

    Returns *(updated_entries, was_inserted)*.
    """
    for i, existing in enumerate(entries):
        if existing.document_id == new_entry.document_id:
            # Non-destructive merge: keep existing value where new_entry has None
            merged = ManifestEntry(
                document_id=new_entry.document_id,
                filename=new_entry.filename or existing.filename,
                standard_number=new_entry.standard_number or existing.standard_number,
                revision=new_entry.revision or existing.revision,
                source_type=new_entry.source_type or existing.source_type,
                source_url=new_entry.source_url or existing.source_url,
                local_path=new_entry.local_path or existing.local_path,
                status=new_entry.status or existing.status,
                downloaded=new_entry.downloaded or existing.downloaded,
                title=new_entry.title or existing.title,
                notes=new_entry.notes or existing.notes,
            )
            entries[i] = merged
            return entries, False
    entries.append(new_entry)
    return entries, True



# ---------------------------------------------------------------------------
# Local PDF Discovery
# ---------------------------------------------------------------------------

# Matches filenames like:
#   694_2010_reff2020.pdf
#   2062_2011_reff2021.pdf
#   3196_1_2013_reaff2024_amd4.pdf  (multi-part number)
#   1417.pdf
#   16102_1_2026.pdf
#
# Capture group 1: IS number portion (may include a part suffix like "16102_1")
# Capture group 2: First 4-digit year encountered after the IS number
_PDF_PATTERN = re.compile(
    r"^(\d+(?:_\d+)?)_(\d{4})(?:_.*)?\.pdf$",
    re.IGNORECASE,
)

# Simpler pattern for files with just a number and no year
_PDF_NO_YEAR_PATTERN = re.compile(r"^(\d+(?:_\d+)?)\.pdf$", re.IGNORECASE)


def _parse_pdf_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Best-effort extraction of (is_number_raw, revision_year) from a PDF filename.

    Returns (None, None) rather than guessing when ambiguous.
    """
    m = _PDF_PATTERN.match(filename)
    if m:
        raw_num = m.group(1)  # e.g. "694" or "16102_1" or "3196_1"
        year = m.group(2)
        # Build IS number string: replace first underscore separator with space
        # for single-part numbers, or use slash for multi-part (e.g. "16102-1")
        parts = raw_num.split("_")
        if len(parts) == 1:
            is_number = f"IS {parts[0]}"
        else:
            # e.g. "3196_1" → "IS 3196 Part 1" — keep it simple as "IS 3196-1"
            is_number = f"IS {parts[0]}-{'-'.join(parts[1:])}"
        return is_number, year

    m2 = _PDF_NO_YEAR_PATTERN.match(filename)
    if m2:
        raw_num = m2.group(1)
        parts = raw_num.split("_")
        if len(parts) == 1:
            is_number = f"IS {parts[0]}"
        else:
            is_number = f"IS {parts[0]}-{'-'.join(parts[1:])}"
        return is_number, None

    return None, None


def _make_document_id(filename: str, is_number: str | None, revision: str | None) -> str:
    """
    Derive a stable, filesystem-safe document ID from what we know.

    Examples:
        694_2010_reff2020.pdf  → bis_is_694_2010
        1417.pdf               → bis_is_1417
        some_weird_name.pdf    → bis_doc_some_weird_name
    """
    if is_number and revision:
        clean = is_number.lower().replace(" ", "_").replace("-", "_")
        return f"bis_{clean}_{revision}"
    if is_number:
        clean = is_number.lower().replace(" ", "_").replace("-", "_")
        return f"bis_{clean}"
    stem = Path(filename).stem.lower()
    safe = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"bis_doc_{safe}"


def discover_local_pdfs(
    bis_docs_dir: str | Path,
    relative_root: str | Path | None = None,
) -> list[ManifestEntry]:
    """
    Scan *bis_docs_dir* and return a ``ManifestEntry`` for each PDF found.

    ``relative_root`` is the base for computing ``local_path``; when omitted
    the directory itself is used.
    """
    docs_dir = Path(bis_docs_dir)
    if not docs_dir.exists():
        return []

    root = Path(relative_root) if relative_root else docs_dir.parent

    entries: list[ManifestEntry] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        filename = pdf.name
        is_number, revision = _parse_pdf_filename(filename)
        doc_id = _make_document_id(filename, is_number, revision)

        # Build a readable title when we have enough info
        title: str | None = None
        if is_number and revision:
            title = f"{is_number}:{revision}"
        elif is_number:
            title = is_number

        try:
            local_path = str(pdf.relative_to(root))
        except ValueError:
            local_path = str(pdf)

        entries.append(
            ManifestEntry(
                document_id=doc_id,
                filename=filename,
                standard_number=is_number,
                revision=revision,
                source_type="standard",
                source_url=None,  # never invent a URL
                local_path=local_path,
                status="available",
                downloaded=False,
                title=title,
            )
        )

    return entries
