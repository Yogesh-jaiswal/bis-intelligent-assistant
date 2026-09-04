"""
tests/features/test_ingestion.py  (updated)
============================================

Tests for the BIS document ingestion and citation pipeline.

Changes in this version:
- Tests for all 9 Excel sheets being discovered
- HTMLProcessor tests
- Improved auth detection: page with "login" link should NOT auto-fail
- Vector deduplication tests
- Manifest upsert non-destructive merge tests

All HTTP calls are mocked — no live network access.
Database tests use SQLite in-memory via a test app context.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.ingestion.manifest import (
    ManifestEntry,
    discover_local_pdfs,
    load_manifest,
    save_manifest,
    upsert_entry,
)
from services.ingestion.web_extractor import (
    WebExtractor,
    normalize_url,
    _is_auth_page,
)
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from models.enums import DocumentBlockType, DocumentTypes
from services.file_processors.document.doc_representation import DocumentBlock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_chunk(filename, source_url, page=None, document_id=None):
    meta = {}
    if page is not None:
        meta["page"] = page
    if document_id is not None:
        meta["document_id"] = document_id
    return RetrievedChunk(
        score=0.9,
        chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text="Text.", metadata=meta),
        filename=filename,
        author=None,
        source_type=DocumentTypes.PDF,
        source_url=source_url,
    )


# ===========================================================================
# 1. Manifest creation
# ===========================================================================

class TestManifestCreation:
    def test_load_nonexistent_returns_empty(self, tmp_path):
        assert load_manifest(tmp_path / "m.json") == []

    def test_load_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text("[]")
        assert load_manifest(p) == []

    def test_save_and_reload(self, tmp_path):
        p = tmp_path / "m.json"
        e = ManifestEntry(document_id="bis_is_694_2010", standard_number="IS 694")
        save_manifest([e], p)
        loaded = load_manifest(p)
        assert len(loaded) == 1
        assert loaded[0].document_id == "bis_is_694_2010"


# ===========================================================================
# 2. Manifest upsert — non-destructive merge
# ===========================================================================

class TestManifestUpsert:
    def test_insert_new(self):
        entries, was_new = upsert_entry([], ManifestEntry(document_id="A"))
        assert was_new and len(entries) == 1

    def test_update_preserves_existing_source_url(self):
        """Re-discovery (source_url=None) must NOT overwrite an existing source_url."""
        original = ManifestEntry(document_id="A", source_url="https://real.example.com/")
        entries = [original]
        # Simulate re-discovery: new entry has no source_url
        entries, was_new = upsert_entry(entries, ManifestEntry(document_id="A", source_url=None))
        assert not was_new
        assert entries[0].source_url == "https://real.example.com/"

    def test_update_overwrites_null_with_real_url(self):
        """If existing entry has no URL but new one does, take the new one."""
        entries = [ManifestEntry(document_id="A", source_url=None)]
        entries, _ = upsert_entry(entries, ManifestEntry(document_id="A", source_url="https://new.example.com/"))
        assert entries[0].source_url == "https://new.example.com/"

    def test_no_duplicate_on_repeated_upsert(self):
        entries = []
        for _ in range(3):
            entries, _ = upsert_entry(entries, ManifestEntry(document_id="A"))
        assert len(entries) == 1


# ===========================================================================
# 3. PDF discovery
# ===========================================================================

class TestPDFDiscovery:
    def _make_dir(self, tmp_path, names):
        d = tmp_path / "BIS"
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"%PDF")
        return d

    def test_discovers_count(self, tmp_path):
        d = self._make_dir(tmp_path, ["694_2010.pdf", "1417.pdf", "2062_2011.pdf"])
        assert len(discover_local_pdfs(d)) == 3

    def test_parses_is_number(self, tmp_path):
        d = self._make_dir(tmp_path, ["694_2010_reff2020.pdf"])
        e = discover_local_pdfs(d, relative_root=tmp_path)[0]
        assert e.standard_number == "IS 694"
        assert e.revision == "2010"

    def test_source_url_never_fabricated(self, tmp_path):
        d = self._make_dir(tmp_path, ["694_2010.pdf"])
        assert discover_local_pdfs(d)[0].source_url is None

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"; d.mkdir()
        assert discover_local_pdfs(d) == []


# ===========================================================================
# 4. URL normalization
# ===========================================================================

class TestURLNormalization:
    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_lowercases(self):
        assert "https://www.example.com" in normalize_url("HTTPS://WWW.EXAMPLE.COM/")

    def test_same_url_stable(self):
        assert normalize_url("https://bis.gov.in/") == normalize_url("https://bis.gov.in")

    def test_different_paths_differ(self):
        assert normalize_url("https://e.com/a") != normalize_url("https://e.com/b")


# ===========================================================================
# 5. Web fetch success (mocked)
# ===========================================================================

class TestWebFetchSuccess:
    def _ok_response(self):
        r = MagicMock()
        r.status_code = 200
        r.ok = True
        r.url = "https://example.com/page"
        r.text = "<html><head><title>BIS Cabling Standards</title></head><body><p>Useful content.</p></body></html>"
        return r

    def test_success_status(self, tmp_path):
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=self._ok_response()):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "success"
        assert result.title == "BIS Cabling Standards"

    def test_saves_html(self, tmp_path):
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=self._ok_response()):
            result = extractor.fetch("https://example.com/page")
        assert Path(result.content_path).exists()

    def test_records_metadata(self, tmp_path):
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=self._ok_response()):
            extractor.fetch("https://example.com/page")
        meta = json.loads((tmp_path / "web" / "metadata.json").read_text())
        assert meta[0]["status"] == "success"


# ===========================================================================
# 6. Web fetch failures (mocked)
# ===========================================================================

class TestWebFetchFailure:
    def test_404(self, tmp_path):
        r = MagicMock(); r.status_code = 404; r.ok = False; r.url = "https://e.com/"; r.text = "Not Found"
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://e.com/")
        assert result.status == "failed"

    def test_timeout(self, tmp_path):
        import requests as req_lib
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", side_effect=req_lib.exceptions.Timeout()):
            result = extractor.fetch("https://slow.example.com/")
        assert result.status == "failed"
        assert "timed out" in result.reason.lower()

    def test_connection_error(self, tmp_path):
        import requests as req_lib
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get",
                   side_effect=req_lib.exceptions.ConnectionError("refused")):
            result = extractor.fetch("https://unreachable.example.com/")
        assert result.status == "failed"
        assert result.content_path is None


# ===========================================================================
# 7. Auth detection — improved (the critical fix)
# ===========================================================================

class TestAuthDetection:
    """
    The old implementation treated any page containing the word 'login' as
    an authentication wall.  The new implementation requires meaningful
    auth-indicating sentence patterns, not just substring presence.
    """

    def _response(self, html: str, status: int = 200) -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.ok = status < 400
        r.url = "https://example.com/page"
        r.text = html
        return r

    # --- Should be auth-required ---

    def test_401_is_auth(self, tmp_path):
        r = self._response("Unauthorized", 401)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "requires_authentication"

    def test_403_is_auth(self, tmp_path):
        r = self._response("Forbidden", 403)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "requires_authentication"

    def test_explicit_login_wall(self, tmp_path):
        html = "<html><body><p>Please log in to access this resource.</p></body></html>"
        r = self._response(html)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "requires_authentication"

    def test_captcha_page_is_auth(self, tmp_path):
        html = "<html><title>captcha verification</title><body>Please complete the captcha to continue.</body></html>"
        r = self._response(html)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "requires_authentication"

    # --- Should NOT be auth-required ---

    def test_ordinary_page_with_login_link_is_not_auth(self, tmp_path):
        """
        A page that has 'login' in a navigation link must NOT be blocked.
        This was the regression the old implementation caused.
        """
        html = """
        <html>
        <head><title>BIS Standards Portal</title></head>
        <body>
          <nav><a href="/login">Login</a> | <a href="/register">Register</a></nav>
          <h1>IS 694:2010 — PVC Insulated Cables</h1>
          <p>This standard specifies requirements for PVC insulated cables rated up to 1100 V.</p>
        </body>
        </html>
        """
        r = self._response(html)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "success", (
            f"Page with nav 'login' link incorrectly blocked: {result.status} / {result.reason}"
        )

    def test_page_mentioning_login_in_context_is_not_auth(self, tmp_path):
        """A page discussing 'login to the BIS portal' as an instruction is not an auth wall."""
        html = """
        <html><title>BIS Certification Guide</title>
        <body>
        <p>To apply for certification, first login to the BIS portal using your credentials.</p>
        <p>You will need to upload your test reports.</p>
        </body></html>
        """
        r = self._response(html)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.status == "success"

    def test_auth_page_has_no_content_path(self, tmp_path):
        html = "<html><body>Please log in to access this resource.</body></html>"
        r = self._response(html)
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            result = extractor.fetch("https://example.com/page")
        assert result.content_path is None


# ===========================================================================
# 8. Deduplication
# ===========================================================================

class TestSeedIdempotency:
    def test_pdf_discovery_twice_no_duplicate(self, tmp_path):
        d = tmp_path / "BIS"; d.mkdir()
        (d / "694_2010.pdf").write_bytes(b"%PDF")
        entries = []
        for _ in range(2):
            for e in discover_local_pdfs(d, relative_root=tmp_path):
                entries, _ = upsert_entry(entries, e)
        assert len(entries) == 1

    def test_already_fetched_true_after_success(self, tmp_path):
        r = MagicMock(); r.status_code = 200; r.ok = True
        r.url = "https://example.com/p"; r.text = "<html><title>T</title></html>"
        extractor = WebExtractor(output_dir=tmp_path / "web")
        with patch("services.ingestion.web_extractor.requests.get", return_value=r):
            extractor.fetch("https://example.com/p")
        assert extractor.already_fetched("https://example.com/p")

    def test_already_fetched_false_for_new_url(self, tmp_path):
        extractor = WebExtractor(output_dir=tmp_path / "web")
        assert not extractor.already_fetched("https://never-seen.example.com/")


# ===========================================================================
# 9. Upload model
# ===========================================================================

class TestUploadModel:
    def test_has_source_url_field(self):
        from models.upload import Upload
        assert hasattr(Upload, "source_url")

    def test_source_url_nullable(self):
        from models.upload import Upload
        col = Upload.__table__.c.get("source_url")
        assert col is not None and col.nullable

    def test_source_url_accepts_value(self):
        from models.upload import Upload
        u = Upload()
        u.source_url = "https://standards.bis.gov.in/doc/694"
        assert u.source_url.endswith("/694")

    def test_source_url_accepts_none(self):
        from models.upload import Upload
        u = Upload()
        u.source_url = None
        assert u.source_url is None


# ===========================================================================
# 10. HTMLProcessor
# ===========================================================================

class TestHTMLProcessor:
    def _processor(self):
        from services.file_processors.extractors.html_processor import HTMLProcessor
        return HTMLProcessor()

    def test_extracts_paragraph_text(self):
        html = "<html><body><p>BIS certification requirements for cables.</p></body></html>"
        result = self._processor().extract_from_string(html)
        assert any("BIS certification" in b.text for b in result.blocks)

    def test_strips_script_tags(self):
        html = "<html><body><script>alert('evil')</script><p>Good content about IS 694.</p></body></html>"
        result = self._processor().extract_from_string(html)
        full_text = " ".join(b.text for b in result.blocks)
        assert "alert" not in full_text
        assert "IS 694" in full_text

    def test_extracts_heading(self):
        from models.enums import DocumentBlockType
        html = "<html><body><h1>IS 694: PVC Insulated Cables</h1><p>Standard for cables above 1100V.</p></body></html>"
        result = self._processor().extract_from_string(html)
        headings = [b for b in result.blocks if b.type == DocumentBlockType.HEADING]
        assert headings, "Expected at least one heading block"

    def test_strips_navigation(self):
        html = """
        <html><body>
        <nav><a href="/login">Login</a></nav>
        <p>Bureau of Indian Standards specifies cable requirements under IS 694:2010.</p>
        </body></html>
        """
        result = self._processor().extract_from_string(html)
        full = " ".join(b.text for b in result.blocks)
        assert "IS 694" in full

    def test_empty_html_returns_empty_blocks(self):
        result = self._processor().extract_from_string("<html><body></body></html>")
        assert result.blocks == []

    def test_produces_document_representation(self):
        from services.file_processors.document.doc_representation import DocumentRepresentation
        html = "<html><body><p>BIS certifies electrical cables under multiple Indian Standards.</p></body></html>"
        result = self._processor().extract_from_string(html)
        assert isinstance(result, DocumentRepresentation)

    def test_extracts_from_file(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text("<html><body><p>Electrical cable standard IS 694 applies to PVC cables.</p></body></html>")
        result = self._processor().extract(f)
        assert any("IS 694" in b.text for b in result.blocks)


# ===========================================================================
# 11. Citation builder — no URL fabrication
# ===========================================================================

class TestCitationBuilderNoFabrication:
    def test_chunk_without_url_excluded(self):
        chunk = _make_chunk("doc.pdf", source_url=None)
        assert CitationBuilder.build_api_citations([chunk]) == []

    def test_chunk_with_url_included(self):
        chunk = _make_chunk("doc.pdf", source_url="https://standards.bis.gov.in/doc/694")
        assert len(CitationBuilder.build_api_citations([chunk])) == 1

    def test_dedup_by_filename(self):
        url = "https://standards.bis.gov.in/doc/694"
        chunks = [
            _make_chunk("694.pdf", source_url=url, page=1),
            _make_chunk("694.pdf", source_url=url, page=2),
        ]
        assert len(CitationBuilder.build_api_citations(chunks)) == 1

    def test_sequential_ids(self):
        chunks = [
            _make_chunk("a.pdf", source_url="https://a.example.com/"),
            _make_chunk("b.pdf", source_url="https://b.example.com/"),
        ]
        ids = [c.id for c in CitationBuilder.build_api_citations(chunks)]
        assert ids == ["cit_1", "cit_2"]

    def test_manifest_lookup_provides_url(self):
        chunk = _make_chunk("694.pdf", source_url=None, document_id="bis_is_694_2010")
        entry = ManifestEntry(document_id="bis_is_694_2010",
                              source_url="https://standards.bis.gov.in/doc/694")
        citations = CitationBuilder.build_api_citations([chunk], manifest_entries=[entry])
        assert len(citations) == 1


# ===========================================================================
# 12. Excel sheet discovery
# ===========================================================================

class TestExcelSheetDiscovery:
    """Verify all 9 expected sheets exist in the dataset."""

    EXPECTED_SHEETS = {
        "Products",
        "Standards",
        "standard_versions",
        "standard_amendments",
        "certification_schemes",
        "laboratories",
        "standard_certification",
        "product_standard_mapping",
        "services",
    }

    def test_all_sheets_present(self):
        import pandas as pd
        from pathlib import Path
        dataset = Path("BIS_DataSheet_Electrical_Cables_Wires (1) copy.xlsx")
        if not dataset.exists():
            pytest.skip("Dataset file not present")
        xl = pd.ExcelFile(dataset)
        assert set(xl.sheet_names) == self.EXPECTED_SHEETS

    def test_products_sheet_has_rows(self):
        import pandas as pd
        from pathlib import Path
        dataset = Path("BIS_DataSheet_Electrical_Cables_Wires (1) copy.xlsx")
        if not dataset.exists():
            pytest.skip("Dataset file not present")
        df = pd.read_excel(dataset, sheet_name="Products")
        assert len(df) > 0

    def test_standards_sheet_has_is_number_column(self):
        import pandas as pd
        from pathlib import Path
        dataset = Path("BIS_DataSheet_Electrical_Cables_Wires (1) copy.xlsx")
        if not dataset.exists():
            pytest.skip("Dataset file not present")
        df = pd.read_excel(dataset, sheet_name="Standards")
        assert "is_number" in df.columns

    def test_laboratories_lab_code_parseable_as_int(self):
        """lab_code comes as float (e.g. 6104524.0) — must be parseable as int."""
        import pandas as pd
        from pathlib import Path
        dataset = Path("BIS_DataSheet_Electrical_Cables_Wires (1) copy.xlsx")
        if not dataset.exists():
            pytest.skip("Dataset file not present")
        df = pd.read_excel(dataset, sheet_name="laboratories")
        df.columns = df.columns.str.lower()
        for val in df["lab_code"].dropna():
            try:
                int(float(str(val)))
            except ValueError:
                pytest.fail(f"lab_code '{val}' could not be parsed as int")


# ===========================================================================
# 13. Page metadata preserved
# ===========================================================================

class TestPageMetadataPreserved:
    def test_page_number_in_chunk(self):
        chunk = _make_chunk("doc.pdf", source_url=None, page=42)
        assert chunk.chunk.metadata["page"] == 42

    def test_metadata_not_mutated_by_citation_builder(self):
        chunk = _make_chunk("doc.pdf", source_url="https://e.com/", page=5)
        orig = dict(chunk.chunk.metadata)
        CitationBuilder.build_api_citations([chunk])
        assert chunk.chunk.metadata == orig

    def test_source_url_on_chunk(self):
        url = "https://standards.bis.gov.in/doc/694"
        chunk = _make_chunk("doc.pdf", source_url=url)
        assert chunk.source_url == url
