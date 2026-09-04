"""
app/commands/seed_dataset.py
============================

CLI command to seed the database from:
  1. Excel dataset (BIS_DataSheet_Electrical_Cables_Wires (1).xlsx)
  2. Local PDFs in BIS_documents/
  3. Authoritative web knowledge in data/web/
  4. Public URLs discovered from the PDF manifest
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import date
from pathlib import Path

import click
import pandas as pd
from flask.cli import with_appcontext
from sqlalchemy import select

from app.extensions import db
from models.certification_scheme import CertificationScheme
from models.laboratory import Laboratory
from models.product import Product
from models.product_standard_mapping import ProductStandardMapping
from models.service import Service
from models.standard import Standard
from models.standard_amendment import StandardAmendment
from models.standard_certification import StandardCertification
from models.standard_version import StandardVersion
from services.ingestion.document_ingestion import ingest_document
from services.ingestion.manifest import (
    ManifestEntry,
    discover_local_pdfs,
    load_manifest,
    save_manifest,
    upsert_entry,
)
from services.ingestion.web_extractor import WebExtractor, normalize_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BIS_DOCS_DIR = _PROJECT_ROOT / "BIS_documents"
_MANIFEST_PATH = _PROJECT_ROOT / "data" / "documents" / "manifest.json"
_WEB_OUTPUT_DIR = _PROJECT_ROOT / "data" / "web"


def _find_dataset_path() -> Path | None:
    """Dynamically resolve the Excel dataset path."""
    candidates = [
        _PROJECT_ROOT / "BIS_DataSheet",
    ]
    for c in candidates:
        if c.exists():
            return c
    xlsx_files = list(_PROJECT_ROOT.glob("*.xlsx"))
    return xlsx_files[0] if xlsx_files else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str | None:
    """Convert a pandas cell to str, returning None for NaN/None/empty."""
    if pd.isna(val) if isinstance(val, float) else val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _safe_int(val) -> int | None:
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def _safe_date(val) -> date | None:
    if pd.isna(val) if isinstance(val, float) else val is None:
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("seed-dataset")
@click.option("--skip-web", is_flag=True, default=False,
              help="Skip web fetch and web vector ingestion.")
@click.option("--skip-uploads", is_flag=True, default=False,
              help="Skip PDF vector ingestion into the vector store.")
@click.option("--fake-embed", is_flag=True, default=False,
              help="Use fake embedder (for CI / dev without a GPU).")
@with_appcontext
def seed_dataset_command(skip_web: bool, skip_uploads: bool, fake_embed: bool) -> None:
    """
    Seed / update the BIS prototype dataset.

    Safe to run multiple times -- all operations are idempotent.
    """
    _banner("BIS Intelligent Assistant -- Dataset Seeder")

    # Ensure all tables exist in PostgreSQL
    click.echo("[>]  Ensuring database schema & tables exist...")
    logger.info("[SEEDER] Ensuring PostgreSQL database schema and tables exist...")
    db.create_all()

    dataset_path = _find_dataset_path()
    if not dataset_path or not dataset_path.exists():
        click.echo(f"  [X]  Dataset file not found in {_PROJECT_ROOT}")
        logger.warning("[SEEDER] Dataset Excel file not found. Structured seeding skipped.")
    else:
        click.echo(f"[>]  Loading Excel dataset: {dataset_path.name}")
        logger.info("[SEEDER] Loading Excel dataset from '%s'...", dataset_path)
        xl = pd.ExcelFile(dataset_path)
        _run_step("1. Products",              lambda: _seed_products(xl))
        _run_step("2. Standards",             lambda: _seed_standards(xl))
        _run_step("3. Certification schemes", lambda: _seed_certification_schemes(xl))
        _run_step("4. Laboratories",          lambda: _seed_laboratories(xl))
        _run_step("5. Services",              lambda: _seed_services(xl))
        _run_step("6. Standard versions",     lambda: _seed_standard_versions(xl))
        _run_step("7. Standard amendments",   lambda: _seed_standard_amendments(xl))
        _run_step("8. Standard certification",lambda: _seed_standard_certification(xl))
        _run_step("9. Product-standard map",  lambda: _seed_product_standard_mapping(xl))

    # ------------------------------------------------------------------
    # PDF manifest + vector ingestion
    # ------------------------------------------------------------------
    click.echo("\n[>]  Step 10: Discover local BIS PDFs")
    logger.info("[SEEDER: STEP 10] Discovering local PDFs in '%s'...", _BIS_DOCS_DIR)
    manifest_entries = load_manifest(_MANIFEST_PATH)
    new_pdfs = updated_pdfs = 0
    for entry in discover_local_pdfs(_BIS_DOCS_DIR, relative_root=_PROJECT_ROOT):
        # Match standard in DB to populate source_url if available
        if entry.standard_number and not entry.source_url:
            clean_num = entry.standard_number.replace("IS", "").strip()
            std_match = db.session.execute(
                select(Standard).where(Standard.is_number.ilike(f"%{clean_num}%"))
            ).scalars().first()
            if std_match and (std_match.document_url or std_match.source_url):
                entry.source_url = std_match.document_url or std_match.source_url

        manifest_entries, was_new = upsert_entry(manifest_entries, entry)
        if was_new:
            new_pdfs += 1
        else:
            updated_pdfs += 1
    save_manifest(manifest_entries, _MANIFEST_PATH)
    click.echo(f"     PDFs discovered: {new_pdfs} new, {updated_pdfs} already in manifest")
    logger.info("[SEEDER: STEP 10] Discovered %d new PDFs, %d existing in manifest", new_pdfs, updated_pdfs)

    _backfill_upload_source_urls()

    if not skip_uploads:
        click.echo("\n[>]  Step 11: Ingest local PDFs into vector store")
        logger.info("[SEEDER: STEP 11] Ingesting local PDFs into vector store (fake_embed=%s)...", fake_embed)
        created = skipped = failed = 0
        for entry in manifest_entries:
            if not entry.local_path:
                continue
            full = _PROJECT_ROOT / entry.local_path
            if not full.exists():
                logger.warning("[SEEDER: STEP 11] Local PDF file missing: %s", full)
                continue
            status, n_chunks = ingest_document(
                file_path=full,
                file_type="pdf",
                filename=entry.filename or full.name,
                source_url=entry.source_url,
                fake_embedder=fake_embed,
            )
            if status == "created":
                created += 1
                click.echo(f"     [OK]  {entry.filename} ({n_chunks} chunks)")
            elif status == "skipped":
                skipped += 1
                click.echo(f"     [SKIP]  {entry.filename} (already ingested)")
            else:
                failed += 1
                click.echo(f"     [X]  {entry.filename} (failed)")
        click.echo(f"     PDFs: {created} ingested, {skipped} skipped, {failed} failed")
        logger.info("[SEEDER: STEP 11] PDF ingestion complete: %d ingested, %d skipped, %d failed", created, skipped, failed)
    else:
        click.echo("\n[>]  Step 11: PDF ingestion skipped (--skip-uploads)")

    # ------------------------------------------------------------------
    # Ingest Curated Local Web Knowledge & Fetch Public Sources
    # ------------------------------------------------------------------
    if not skip_uploads and _WEB_OUTPUT_DIR.exists():
        click.echo("\n[>]  Step 12: Ingest authoritative BIS web knowledge documents")
        logger.info("[SEEDER: STEP 12] Ingesting authoritative local web knowledge from '%s'...", _WEB_OUTPUT_DIR)
        web_created = web_skipped = 0
        extractor = WebExtractor(output_dir=_WEB_OUTPUT_DIR)
        for html_file in _WEB_OUTPUT_DIR.glob("*.html"):
            source_id = html_file.stem
            records = json.loads((_WEB_OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8")) if (_WEB_OUTPUT_DIR / "metadata.json").exists() else []
            source_url = next((r.get("original_url") for r in records if r.get("source_id") == source_id), "https://www.bis.gov.in/")
            status, n_chunks = ingest_document(
                file_path=html_file,
                file_type="html",
                filename=html_file.name,
                source_url=source_url,
                fake_embedder=fake_embed,
            )
            if status == "created":
                web_created += 1
                click.echo(f"     [OK]  {html_file.name} ({n_chunks} chunks)")
            elif status == "skipped":
                web_skipped += 1
                click.echo(f"     [SKIP]  {html_file.name} (already ingested)")
        click.echo(f"     Authoritative Web Docs: {web_created} ingested, {web_skipped} skipped")
        logger.info("[SEEDER: STEP 12] Web knowledge ingestion complete: %d ingested, %d skipped", web_created, web_skipped)

    if skip_web:
        click.echo("\n[>]  Step 13: Live web fetch skipped (--skip-web)")
    else:
        click.echo("\n[>]  Step 13: Fetch & ingest public web sources")
        logger.info("[SEEDER: STEP 13] Fetching public web sources from manifest entries...")
        urls_to_fetch = sorted(set(
            e.source_url for e in manifest_entries if e.source_url
        ))
        w_fetched = w_skipped = w_failed = w_auth = w_vec_created = w_vec_skipped = 0
        extractor = WebExtractor(output_dir=_WEB_OUTPUT_DIR)

        for url in urls_to_fetch:
            if extractor.already_fetched(url):
                w_skipped += 1
                click.echo(f"     [SKIP]  {url[:80]}  (already fetched)")
                content_path = extractor.content_path_for_url(url)
                if content_path and not skip_uploads:
                    source_id = extractor.source_id_for_url(url)
                    status, _ = ingest_document(
                        file_path=content_path,
                        file_type="html",
                        filename=source_id + ".html",
                        source_url=url,
                        fake_embedder=fake_embed,
                    )
                    if status == "skipped":
                        w_vec_skipped += 1
                continue

            click.echo(f"     ->  {url[:80]}")
            result = extractor.fetch(url)

            if result.status == "success":
                w_fetched += 1
                click.echo(f"        [OK]  HTTP {result.http_status}  {result.title or ''}")
                if not skip_uploads and result.content_path:
                    status, n = ingest_document(
                        file_path=result.content_path,
                        file_type="html",
                        filename=result.source_id + ".html",
                        source_url=url,
                        fake_embedder=fake_embed,
                    )
                    if status == "created":
                        w_vec_created += 1
                        click.echo(f"           -> {n} chunks ingested")
                    elif status == "skipped":
                        w_vec_skipped += 1
            elif result.status == "requires_authentication":
                w_auth += 1
                click.echo(f"        [AUTH]  Requires authentication -- skipped")
                logger.info("[SEEDER: STEP 13] URL requires authentication (skipped): %s", url)
            else:
                w_failed += 1
                click.echo(f"        [X]  {result.reason}")
                logger.warning("[SEEDER: STEP 13] Failed to fetch URL %s: %s", url, result.reason)

        click.echo(f"     URLs: {w_fetched} fetched, {w_skipped} skipped, {w_failed} failed, {w_auth} auth-blocked")
        click.echo(f"     Web vectors: {w_vec_created} ingested, {w_vec_skipped} skipped")
        logger.info("[SEEDER: STEP 13] Web fetch complete: %d fetched, %d skipped, %d failed, %d auth-blocked", w_fetched, w_skipped, w_failed, w_auth)

    _banner("Seeding complete")
    logger.info("[SEEDER: COMPLETE] Dataset seeding execution completed successfully.")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    click.echo(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def _run_step(label: str, fn) -> None:
    click.echo(f"\n[>]  Step {label}")
    logger.info("[SEEDER: STEP START] Running Step %s...", label)
    try:
        inserted, updated, failed = fn()
        click.echo(f"     {inserted} inserted, {updated} updated, {failed} failed")
        logger.info("[SEEDER: STEP RESULT] Step %s -> %d inserted, %d updated, %d failed", label, inserted, updated, failed)
    except Exception as exc:
        click.echo(f"     [X]  Step failed: {exc}")
        logger.error("[SEEDER: STEP ERROR] Step %s failed with exception: %s", label, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Sheet seeders
# ---------------------------------------------------------------------------

def _seed_products(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("Products")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    for _, row in df.iterrows():
        try:
            name = _safe_str(row.get("name"))
            if not name:
                fail += 1
                logger.warning("[SEED: Products] Missing name in row %s", dict(row))
                continue
            obj = db.session.execute(
                select(Product).where(Product.name == name)
            ).scalar_one_or_none()
            if obj is None:
                db.session.add(Product(
                    name=name,
                    category=_safe_str(row.get("category")),
                    description=_safe_str(row.get("description")),
                    keywords=_safe_str(row.get("keywords")),
                ))
                ins += 1
            else:
                obj.category    = _safe_str(row.get("category"))    or obj.category
                obj.description = _safe_str(row.get("description")) or obj.description
                obj.keywords    = _safe_str(row.get("keywords"))     or obj.keywords
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: Products] Error seeding product '%s': %s", row.get("name"), traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_standards(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("Standards")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    for _, row in df.iterrows():
        try:
            is_number = _safe_str(row.get("is_number"))
            if not is_number:
                fail += 1
                logger.warning("[SEED: Standards] Missing is_number in row: %s", dict(row))
                continue
            obj = db.session.execute(
                select(Standard).where(Standard.is_number == is_number)
            ).scalar_one_or_none()
            if obj is None:
                db.session.add(Standard(
                    is_number=is_number,
                    title=_safe_str(row.get("title")) or is_number,
                    revision_no=_safe_int(row.get("revision_no")),
                    publication_year=_safe_int(row.get("publication_year")),
                    status=_safe_str(row.get("status")),
                    technical_department=_safe_str(row.get("technical_department")),
                    source_url=_safe_str(row.get("source_url")),
                    document_url=_safe_str(row.get("document_url")),
                    last_verified_at=_safe_date(row.get("last_verified_at")),
                ))
                ins += 1
            else:
                obj.title               = _safe_str(row.get("title"))               or obj.title
                obj.revision_no         = _safe_int(row.get("revision_no"))         or obj.revision_no
                obj.publication_year    = _safe_int(row.get("publication_year"))    or obj.publication_year
                obj.status              = _safe_str(row.get("status"))              or obj.status
                obj.technical_department= _safe_str(row.get("technical_department"))or obj.technical_department
                obj.source_url          = _safe_str(row.get("source_url"))          or obj.source_url
                obj.document_url        = _safe_str(row.get("document_url"))        or obj.document_url
                obj.last_verified_at    = _safe_date(row.get("last_verified_at"))   or obj.last_verified_at
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: Standards] Error seeding standard '%s': %s", row.get("is_number"), traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_certification_schemes(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("certification_schemes")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    for _, row in df.iterrows():
        try:
            scheme_code = _safe_str(row.get("scheme_code"))
            if not scheme_code:
                fail += 1
                continue
            obj = db.session.execute(
                select(CertificationScheme).where(CertificationScheme.scheme_code == scheme_code)
            ).scalar_one_or_none()
            if obj is None:
                db.session.add(CertificationScheme(
                    name=_safe_str(row.get("name")) or scheme_code,
                    scheme_code=scheme_code,
                    description=_safe_str(row.get("description")),
                    certification_type=_safe_str(row.get("certification_type")),
                    mandatory=_safe_str(row.get("mandatory")),
                    authority=_safe_str(row.get("authority")),
                    source_url=_safe_str(row.get("source_url")),
                ))
                ins += 1
            else:
                obj.name               = _safe_str(row.get("name"))               or obj.name
                obj.description        = _safe_str(row.get("description"))        or obj.description
                obj.certification_type = _safe_str(row.get("certification_type")) or obj.certification_type
                obj.mandatory          = _safe_str(row.get("mandatory"))          or obj.mandatory
                obj.authority          = _safe_str(row.get("authority"))          or obj.authority
                obj.source_url         = _safe_str(row.get("source_url"))         or obj.source_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: CertificationSchemes] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_laboratories(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("laboratories")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    for _, row in df.iterrows():
        try:
            name = _safe_str(row.get("name"))
            if not name:
                fail += 1
                continue
            raw_code = row.get("lab_code")
            lab_code: str | None = None
            if raw_code is not None and not (isinstance(raw_code, float) and pd.isna(raw_code)):
                lab_code = str(int(float(str(raw_code))))

            obj = db.session.execute(
                select(Laboratory).where(Laboratory.name == name)
            ).scalar_one_or_none()
            if obj is None:
                db.session.add(Laboratory(
                    lab_code=lab_code,
                    name=name,
                    address=_safe_str(row.get("address")),
                    state=_safe_str(row.get("state")),
                    district=_safe_str(row.get("district")),
                    contact_person=_safe_str(row.get("contact_person")),
                    phone=_safe_str(row.get("phone")),
                    email=_safe_str(row.get("email")),
                    validity_date=_safe_date(row.get("validity_date")),
                    scope=_safe_str(row.get("scope")),
                    source_url=_safe_str(row.get("source_url")),
                ))
                ins += 1
            else:
                obj.lab_code      = lab_code          or obj.lab_code
                obj.address       = _safe_str(row.get("address"))       or obj.address
                obj.state         = _safe_str(row.get("state"))         or obj.state
                obj.district      = _safe_str(row.get("district"))      or obj.district
                obj.contact_person= _safe_str(row.get("contact_person"))or obj.contact_person
                obj.phone         = _safe_str(row.get("phone"))         or obj.phone
                obj.email         = _safe_str(row.get("email"))         or obj.email
                obj.validity_date = _safe_date(row.get("validity_date"))or obj.validity_date
                obj.scope         = _safe_str(row.get("scope"))         or obj.scope
                obj.source_url    = _safe_str(row.get("source_url"))    or obj.source_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: Laboratories] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_services(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("services")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    for _, row in df.iterrows():
        try:
            name = _safe_str(row.get("name"))
            if not name:
                fail += 1
                continue
            obj = db.session.execute(
                select(Service).where(Service.name == name)
            ).scalar_one_or_none()
            if obj is None:
                db.session.add(Service(
                    name=name,
                    service_type=_safe_str(row.get("service_type")) or "General",
                    description=_safe_str(row.get("description")),
                    eligibility=_safe_str(row.get("eligibility")),
                    documents_required=_safe_str(row.get("documents_required")),
                    source_url=_safe_str(row.get("source_url")),
                ))
                ins += 1
            else:
                obj.service_type       = _safe_str(row.get("service_type"))       or obj.service_type
                obj.description        = _safe_str(row.get("description"))        or obj.description
                obj.eligibility        = _safe_str(row.get("eligibility"))        or obj.eligibility
                obj.documents_required = _safe_str(row.get("documents_required")) or obj.documents_required
                obj.source_url         = _safe_str(row.get("source_url"))         or obj.source_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: Services] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_standard_versions(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("standard_versions")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    standards_by_excel_id = _load_excel_id_map(Standard, "standards")

    for idx, row in df.iterrows():
        try:
            excel_std_id = _safe_int(row.get("standard_id"))
            if excel_std_id is None:
                fail += 1
                continue
            std = standards_by_excel_id.get(excel_std_id)
            if std is None:
                fail += 1
                logger.warning("[SEED: StandardVersions] Row %d: standard_id %s not found in DB mapping", idx + 1, excel_std_id)
                continue

            version = _safe_str(row.get("version"))
            if not version:
                fail += 1
                continue

            obj = db.session.execute(
                select(StandardVersion).where(
                    StandardVersion.standard_id == std.id,
                    StandardVersion.version == version,
                )
            ).scalar_one_or_none()

            if obj is None:
                db.session.add(StandardVersion(
                    standard_id=std.id,
                    version=version,
                    version_type=_safe_str(row.get("version_type")),
                    publication_date=_safe_date(row.get("publication_date")),
                    effective_date=_safe_date(row.get("effective_date")),
                    status=_safe_str(row.get("status")),
                    document_url=_safe_str(row.get("document_url")),
                ))
                ins += 1
            else:
                obj.version_type     = _safe_str(row.get("version_type"))     or obj.version_type
                obj.publication_date = _safe_date(row.get("publication_date")) or obj.publication_date
                obj.effective_date   = _safe_date(row.get("effective_date"))   or obj.effective_date
                obj.status           = _safe_str(row.get("status"))           or obj.status
                obj.document_url     = _safe_str(row.get("document_url"))     or obj.document_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: StandardVersions] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_standard_amendments(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("standard_amendments")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    standards_by_excel_id = _load_excel_id_map(Standard, "standards")

    for idx, row in df.iterrows():
        try:
            excel_std_id = _safe_int(row.get("standard_id"))
            amendment_number = _safe_str(row.get("amendment_number"))
            if excel_std_id is None or amendment_number is None:
                fail += 1
                continue
            std = standards_by_excel_id.get(excel_std_id)
            if std is None:
                fail += 1
                logger.warning("[SEED: StandardAmendments] Row %d: standard_id %s not found in DB mapping", idx + 1, excel_std_id)
                continue

            obj = db.session.execute(
                select(StandardAmendment).where(
                    StandardAmendment.standard_id == std.id,
                    StandardAmendment.amendment_number == amendment_number,
                )
            ).scalar_one_or_none()

            if obj is None:
                db.session.add(StandardAmendment(
                    standard_id=std.id,
                    amendment_number=amendment_number,
                    title=_safe_str(row.get("title")),
                    publication_date=_safe_date(row.get("publication_date")),
                    effective_date=_safe_date(row.get("effective_date")),
                    document_url=_safe_str(row.get("document_url")),
                ))
                ins += 1
            else:
                obj.title            = _safe_str(row.get("title"))            or obj.title
                obj.publication_date = _safe_date(row.get("publication_date")) or obj.publication_date
                obj.effective_date   = _safe_date(row.get("effective_date"))   or obj.effective_date
                obj.document_url     = _safe_str(row.get("document_url"))     or obj.document_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: StandardAmendments] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_standard_certification(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("standard_certification")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    standards_by_excel_id = _load_excel_id_map(Standard, "standards")
    schemes_by_excel_id   = _load_excel_id_map(CertificationScheme, "certification_schemes")

    for idx, row in df.iterrows():
        try:
            excel_std_id    = _safe_int(row.get("standard_id"))
            excel_scheme_id = _safe_int(row.get("certification_scheme_id"))
            if excel_std_id is None or excel_scheme_id is None:
                fail += 1
                continue
            std    = standards_by_excel_id.get(excel_std_id)
            scheme = schemes_by_excel_id.get(excel_scheme_id)
            if std is None or scheme is None:
                fail += 1
                logger.warning("[SEED: StandardCertification] Row %d: standard_id %s or scheme_id %s not found in DB mapping", idx + 1, excel_std_id, excel_scheme_id)
                continue

            obj = db.session.execute(
                select(StandardCertification).where(
                    StandardCertification.standard_id == std.id,
                    StandardCertification.certification_scheme_id == scheme.id,
                )
            ).scalar_one_or_none()

            if obj is None:
                db.session.add(StandardCertification(
                    standard_id=std.id,
                    certification_scheme_id=scheme.id,
                    requirement_type=_safe_str(row.get("requirement_type")),
                    mandatory=_safe_str(row.get("mandatory")),
                    conditions=_safe_str(row.get("conditions")),
                    source_url=_safe_str(row.get("source_url")),
                ))
                ins += 1
            else:
                obj.requirement_type = _safe_str(row.get("requirement_type")) or obj.requirement_type
                obj.mandatory        = _safe_str(row.get("mandatory"))        or obj.mandatory
                obj.conditions       = _safe_str(row.get("conditions"))       or obj.conditions
                obj.source_url       = _safe_str(row.get("source_url"))       or obj.source_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: StandardCertification] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


def _seed_product_standard_mapping(xl: pd.ExcelFile) -> tuple[int, int, int]:
    df = xl.parse("product_standard_mapping")
    df.columns = df.columns.str.lower()
    ins = upd = fail = 0
    products_by_excel_id  = _load_excel_id_map(Product, "products")
    standards_by_excel_id = _load_excel_id_map(Standard, "standards")

    for idx, row in df.iterrows():
        try:
            excel_prod_id = _safe_int(row.get("product_id"))
            excel_std_id  = _safe_int(row.get("standard_id"))
            if excel_prod_id is None or excel_std_id is None:
                fail += 1
                continue
            prod = products_by_excel_id.get(excel_prod_id)
            std  = standards_by_excel_id.get(excel_std_id)
            if prod is None or std is None:
                fail += 1
                logger.warning("[SEED: ProductStandardMapping] Row %d: product_id %s or standard_id %s not found in DB mapping", idx + 1, excel_prod_id, excel_std_id)
                continue

            obj = db.session.execute(
                select(ProductStandardMapping).where(
                    ProductStandardMapping.product_id == prod.id,
                    ProductStandardMapping.standard_id == std.id,
                )
            ).scalar_one_or_none()

            if obj is None:
                db.session.add(ProductStandardMapping(
                    product_id=prod.id,
                    standard_id=std.id,
                    relevance=_safe_str(row.get("relevance")),
                    source_url=_safe_str(row.get("source_url")),
                ))
                ins += 1
            else:
                obj.relevance  = _safe_str(row.get("relevance"))  or obj.relevance
                obj.source_url = _safe_str(row.get("source_url")) or obj.source_url
                upd += 1
        except Exception:
            fail += 1
            logger.error("[SEED: ProductStandardMapping] Error: %s", traceback.format_exc())
            db.session.rollback()
    _commit()
    return ins, upd, fail


# ---------------------------------------------------------------------------
# FK resolution helper
# ---------------------------------------------------------------------------

_EXCEL_ID_CACHE: dict[str, dict[int, object]] = {}


def _load_excel_id_map(model_class, cache_key: str) -> dict[int, object]:
    """
    Build a {excel_row_id → ORM object} mapping.
    """
    if cache_key in _EXCEL_ID_CACHE:
        return _EXCEL_ID_CACHE[cache_key]

    rows = db.session.execute(
        select(model_class).order_by(model_class.id)
    ).scalars().all()

    mapping: dict[int, object] = {}
    for row in rows:
        mapping[row.id] = row

    _EXCEL_ID_CACHE[cache_key] = mapping
    return mapping


def _commit() -> None:
    """Commit the current transaction, rolling back on error."""
    try:
        db.session.commit()
        _EXCEL_ID_CACHE.clear()
    except Exception as exc:
        logger.error("[SEEDER: DB COMMIT ERROR] Commit failed: %s", exc, exc_info=True)
        db.session.rollback()


def _backfill_upload_source_urls() -> int:
    """Backfill missing source_url on Upload records by matching standard is_number."""
    try:
        from models.upload import Upload
        from models.standard import Standard
        uploads = db.session.execute(select(Upload)).scalars().all()
        standards = db.session.execute(select(Standard)).scalars().all()
        std_map = {}
        for std in standards:
            if std.is_number:
                clean = std.is_number.replace("IS", "").strip()
                num_part = clean.split(":")[0].split("-")[0].strip()
                if num_part:
                    std_map[num_part] = std.document_url or std.source_url

        updated = 0
        for up in uploads:
            if not up.source_url and up.filename:
                fname = up.filename
                for num_part, url in std_map.items():
                    if num_part in fname:
                        up.source_url = url
                        updated += 1
                        break
        if updated > 0:
            db.session.commit()
            logger.info("[SEEDER] Backfilled source_url for %d Upload records", updated)
        return updated
    except Exception as exc:
        logger.warning("[SEEDER] Upload source_url backfill skipped: %s", exc)
        return 0
