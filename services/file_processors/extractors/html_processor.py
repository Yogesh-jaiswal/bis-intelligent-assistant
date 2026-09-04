"""
HTML document processor for the BIS Intelligent Assistant.

Extracts visible, meaningful text from saved HTML pages using BeautifulSoup.

What is stripped:
- <script> and <style> tags
- <nav>, <footer>, <header> navigation blocks
- Cookie banners (common class patterns)
- HTML markup and attributes

What is kept:
- Headings (mapped to HEADING block type)
- Paragraphs and list items (PARAGRAPH)
- Tables (TABLE, rendered as pipe-delimited rows)

This processor integrates with the existing ``DocumentProcessorFactory``
and ``FileProcessor`` pipeline so web content goes through the same
chunking → embedding → storage path as PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from .base_processor import BaseProcessor
from models.enums import DocumentBlockType
from services.file_processors.document.doc_representation import (
    DocumentBlock,
    DocumentRepresentation,
)

# ---------------------------------------------------------------------------
# Tags / CSS classes to strip entirely
# ---------------------------------------------------------------------------

_STRIP_TAGS = {
    "script", "style", "noscript",
    "nav", "footer", "header",
    "aside", "form", "button", "input", "select",
    "iframe", "svg", "canvas",
}

_STRIP_CLASS_FRAGMENTS = {
    "cookie", "banner", "popup", "modal", "overlay",
    "newsletter", "subscribe", "ad", "advertisement",
}

# Heading tag names → HEADING block type
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Tags whose text becomes PARAGRAPH blocks
_PARAGRAPH_TAGS = {"p", "li", "dd", "blockquote", "figcaption", "td", "th"}

# Minimum character count to keep a general paragraph text block
_MIN_PARAGRAPH_LENGTH = 10
_MIN_HEADING_LENGTH = 3


class HTMLProcessor(BaseProcessor):
    """
    Extracts visible text content from an HTML file and returns a
    ``DocumentRepresentation`` compatible with the existing chunker/
    embedding pipeline.
    """

    def extract(self, file_path: str | Path) -> DocumentRepresentation:
        path = Path(file_path)
        html = path.read_text(encoding="utf-8", errors="replace")
        return self._parse(html, source_label=path.name)

    def extract_from_string(self, html: str, source_label: str = "web") -> DocumentRepresentation:
        """Parse HTML given as a string (useful for direct WebExtractor integration)."""
        return self._parse(html, source_label=source_label)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, html: str, source_label: str) -> DocumentRepresentation:
        soup = BeautifulSoup(html, "lxml")

        # Remove noisy tags
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Remove elements with cookie/ad class patterns
        for tag in soup.find_all(True):
            cls = " ".join(tag.get("class", [])).lower()
            if any(frag in cls for frag in _STRIP_CLASS_FRAGMENTS):
                tag.decompose()

        # Extract title for author/provenance
        title_tag = soup.find("title")
        doc_title = title_tag.get_text(strip=True) if title_tag else None

        # Walk the body tree and collect blocks
        body = soup.find("body") or soup
        blocks = self._collect_blocks(body)

        # Deduplicate by text
        seen: set[str] = set()
        unique_blocks: list[DocumentBlock] = []
        for block in blocks:
            text = block.text.strip()
            min_len = _MIN_HEADING_LENGTH if block.type == DocumentBlockType.HEADING else _MIN_PARAGRAPH_LENGTH
            if not text or text in seen or len(text) < min_len:
                continue
            seen.add(text)
            block.text = text
            unique_blocks.append(block)

        return DocumentRepresentation(
            author=None,
            blocks=unique_blocks,
        )


    def _collect_blocks(self, root: Tag) -> list[DocumentBlock]:
        """Walk the tag tree and emit DocumentBlocks for headings, paragraphs, and tables."""
        blocks: list[DocumentBlock] = []

        for element in root.descendants:
            if not isinstance(element, Tag):
                continue

            tag_name = element.name.lower() if element.name else ""

            if tag_name in _HEADING_TAGS:
                text = element.get_text(" ", strip=True)
                if text:
                    blocks.append(DocumentBlock(
                        type=DocumentBlockType.HEADING,
                        text=text,
                        metadata={"html_tag": tag_name},
                    ))

            elif tag_name == "table":
                table_text = self._extract_table(element)
                if table_text:
                    blocks.append(DocumentBlock(
                        type=DocumentBlockType.TABLE,
                        text=table_text,
                        metadata={},
                    ))

            elif tag_name in _PARAGRAPH_TAGS:
                # Skip if this element is inside a table (already extracted above)
                if element.find_parent("table"):
                    continue
                text = element.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    blocks.append(DocumentBlock(
                        type=DocumentBlockType.PARAGRAPH,
                        text=text,
                        metadata={},
                    ))

        return blocks

    @staticmethod
    def _extract_table(table_tag: Tag) -> str:
        """Render an HTML table as pipe-delimited text rows."""
        rows = []
        for row in table_tag.find_all("tr"):
            cells = row.find_all(["td", "th"])
            row_text = " | ".join(
                re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
                for cell in cells
            )
            if row_text.strip():
                rows.append(row_text)
        return "\n".join(rows)
