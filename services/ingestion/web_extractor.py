"""
Basic public-web extractor for the BIS Intelligent Assistant.

This is a controlled fetcher for URLs already present in our dataset,
not a general-purpose crawler.

Rules enforced here:
- Only fetch publicly accessible URLs (normal HTTP/HTTPS).
- Respect normal HTTP behaviour; no CAPTCHA bypass.
- Do not authenticate to any service.
- Reasonable per-request timeout.
- No aggressive crawling — one URL at a time, caller controls loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FETCH_TIMEOUT_SECONDS: int = 15
"""Hard timeout for each HTTP request."""

_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "BIS-Intelligent-Assistant/1.0 "
        "(SIH prototype; +https://github.com/bis-intelligent-assistant)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

FetchStatus = Literal["success", "failed", "requires_authentication", "skipped"]

# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------

# Patterns that strongly indicate a login/auth wall.
# We match against <title> and visible body text (not arbitrary HTML content).
# A pattern like r'\blogin\b' only fires when "login" is a standalone word in
# the small text sample — not when it appears inside URLs or class names.
_AUTH_TITLE_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\blogin\b",
        r"\bsign[\s\-]?in\b",
        r"\bsign[\s\-]?up\b",
        r"authentication required",
        r"members only",
        r"access denied",
        r"please log in",
        r"captcha",
        r"verify you are human",
    ]
)

_AUTH_BODY_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        # These patterns are more specific than just "login" because body text
        # can legitimately contain the word "login" in context (e.g. "to login
        # to your BIS portal, visit...").
        r"please (log|sign) in to (access|continue|view)",
        r"you (must|need to) (be logged in|log in|sign in)",
        r"this (page|content|resource) requires (authentication|login|a login)",
        r"enter your (username|email) and password",
        r"forgot your password",
        r"create an account",
        r"verify you['']?re (human|not a robot)",
        r"\bcaptcha\b",
    ]
)


def _is_auth_page(response: requests.Response) -> bool:
    """
    Determine whether a response looks like an authentication wall.

    Strategy:
    1. HTTP 401 or 403 → definitely auth.
    2. Check the page <title> against _AUTH_TITLE_PATTERNS (high precision).
    3. Extract visible text from the body and check against _AUTH_BODY_PATTERNS
       (more specific than simple substring matching).

    A page that merely *mentions* the word "login" (e.g. "visit our login
    portal for account services") is NOT classified as auth-required.
    """
    if response.status_code in (401, 403):
        return True

    try:
        soup = BeautifulSoup(response.text, "lxml")

        # Check title
        title_tag = soup.find("title")
        title_text = (title_tag.get_text(strip=True) if title_tag else "") or ""
        for pattern in _AUTH_TITLE_PATTERNS:
            if pattern.search(title_text):
                return True

        # Check first 2000 chars of visible body text (not raw HTML)
        body = soup.find("body")
        if body:
            visible_text = body.get_text(" ", strip=True)[:2000]
            for pattern in _AUTH_BODY_PATTERNS:
                if pattern.search(visible_text):
                    return True

    except Exception:
        # Parsing failed — fall back to status code only
        pass

    return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WebFetchResult:
    """Metadata record for one URL fetch attempt."""

    source_id: str
    """Stable identifier derived from the URL."""

    original_url: str
    """URL as given by the caller."""

    final_url: str | None
    """URL after any HTTP redirects."""

    fetched_at: str
    """ISO-8601 UTC timestamp."""

    status: FetchStatus

    http_status: int | None
    """HTTP status code, or None on connection error."""

    title: str | None
    """HTML <title> text when successfully parsed."""

    content_path: str | None
    """Relative path to saved HTML file, e.g. 'data/web/<source_id>.html'."""

    reason: str | None
    """Human-readable failure reason."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WebFetchResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """
    Return a normalised form of *url* used for deduplication.

    Normalisation:
    - Lowercases scheme and host.
    - Strips trailing slashes from the path.
    - Does NOT strip query/fragment (different queries may differ).
    """
    try:
        parsed = urlparse(url.strip())
        normalised = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=parsed.path.rstrip("/") or "/",
        )
        return urlunparse(normalised)
    except Exception:
        return url.strip()


def _source_id_from_url(url: str) -> str:
    """
    Derive a stable, filesystem-safe identifier from *url*.

    Uses the first 12 hex chars of the SHA-256 of the normalised URL.
    """
    return "web_" + hashlib.sha256(normalize_url(url).encode()).hexdigest()[:12]


def _extract_title(html: str) -> str | None:
    """Extract the ``<title>`` text from *html*, or ``None``."""
    try:
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("title")
        if tag and tag.string:
            return tag.string.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _load_web_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        return []
    raw = metadata_path.read_text(encoding="utf-8").strip()
    if not raw or raw == "[]":
        return []
    return json.loads(raw)


def _save_web_metadata(records: list[dict], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class WebExtractor:
    """
    Controlled fetcher for publicly accessible URLs already present in our dataset.

    Usage::

        extractor = WebExtractor(output_dir="data/web")
        result = extractor.fetch("https://example.com/some-page")
        print(result.status)  # "success" | "failed" | "requires_authentication"
    """

    def __init__(self, output_dir: str | Path = "data/web") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def already_fetched(self, url: str) -> bool:
        """Return True when *url* (normalised) has already been successfully fetched."""
        norm = normalize_url(url)
        records = _load_web_metadata(self.metadata_path)
        for rec in records:
            if normalize_url(rec.get("original_url", "")) == norm:
                if rec.get("status") == "success":
                    return True
        return False

    def source_id_for_url(self, url: str) -> str:
        """Return the stable source_id for *url* (same logic as internal _source_id_from_url)."""
        return _source_id_from_url(url)

    def content_path_for_url(self, url: str) -> Path | None:
        """
        Return the path to the saved HTML file for *url* if it was previously
        successfully fetched, otherwise None.
        """
        records = _load_web_metadata(self.metadata_path)
        norm = normalize_url(url)
        for rec in records:
            if normalize_url(rec.get("original_url", "")) == norm and rec.get("status") == "success":
                cp = rec.get("content_path")
                if cp:
                    return Path(cp)
        return None

    def fetch(self, url: str) -> WebFetchResult:
        """
        Attempt to fetch *url*.

        Returns a ``WebFetchResult`` describing the outcome.
        Never raises — failures are captured in the result.
        """
        source_id = _source_id_from_url(url)
        fetched_at = datetime.now(timezone.utc).isoformat()

        try:
            response = requests.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
        except requests.exceptions.Timeout:
            result = WebFetchResult(
                source_id=source_id,
                original_url=url,
                final_url=None,
                fetched_at=fetched_at,
                status="failed",
                http_status=None,
                title=None,
                content_path=None,
                reason=f"Request timed out after {FETCH_TIMEOUT_SECONDS}s",
            )
            self._record(result)
            return result

        except requests.exceptions.ConnectionError as exc:
            result = WebFetchResult(
                source_id=source_id,
                original_url=url,
                final_url=None,
                fetched_at=fetched_at,
                status="failed",
                http_status=None,
                title=None,
                content_path=None,
                reason=f"Connection error: {exc}",
            )
            self._record(result)
            return result

        except Exception as exc:
            result = WebFetchResult(
                source_id=source_id,
                original_url=url,
                final_url=None,
                fetched_at=fetched_at,
                status="failed",
                http_status=None,
                title=None,
                content_path=None,
                reason=f"Unexpected error: {exc}",
            )
            self._record(result)
            return result

        final_url = response.url

        # -- Authentication / CAPTCHA check (improved) --
        if _is_auth_page(response):
            result = WebFetchResult(
                source_id=source_id,
                original_url=url,
                final_url=final_url,
                fetched_at=fetched_at,
                status="requires_authentication",
                http_status=response.status_code,
                title=None,
                content_path=None,
                reason="Page appears to require authentication or CAPTCHA",
            )
            self._record(result)
            return result

        # -- HTTP error --
        if not response.ok:
            result = WebFetchResult(
                source_id=source_id,
                original_url=url,
                final_url=final_url,
                fetched_at=fetched_at,
                status="failed",
                http_status=response.status_code,
                title=None,
                content_path=None,
                reason=f"HTTP {response.status_code}",
            )
            self._record(result)
            return result

        # -- Success: save HTML --
        html = response.text
        title = _extract_title(html)
        content_filename = f"{source_id}.html"
        content_file = self.output_dir / content_filename
        content_file.write_text(html, encoding="utf-8", errors="replace")
        content_path = str(self.output_dir / content_filename)

        result = WebFetchResult(
            source_id=source_id,
            original_url=url,
            final_url=final_url,
            fetched_at=fetched_at,
            status="success",
            http_status=response.status_code,
            title=title,
            content_path=content_path,
            reason=None,
        )
        self._record(result)
        logger.info(
            "WebExtractor: fetched %s → %s (HTTP %s)",
            url, final_url, response.status_code,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, result: WebFetchResult) -> None:
        """Upsert *result* in the metadata JSON (keyed by source_id)."""
        records = _load_web_metadata(self.metadata_path)

        # Replace existing entry for the same source_id
        for i, rec in enumerate(records):
            if rec.get("source_id") == result.source_id:
                records[i] = result.to_dict()
                break
        else:
            records.append(result.to_dict())

        _save_web_metadata(records, self.metadata_path)
