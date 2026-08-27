from pathlib import Path

import pymupdf
from PIL import Image

from services.integrations.ocr_service import OCR

from .base_processor import BaseProcessor

from models.enums import DocumentBlockType

from services.file_processors.document.doc_representation import (
    DocumentBlock,
    DocumentRepresentation
)

MIN_TEXT_THRESHOLD = 120
MIN_IMAGE_COVERAGE = 0.7
OCR_DPI = 250

class PDFProcessor(BaseProcessor):
    """PDF file processor"""
    def __init__(self):
        self.ocr = OCR()
        
    def extract(self, file_path: str | Path) -> DocumentRepresentation:
        blocks = []
        seen: set[str] = set()

        with pymupdf.open(file_path) as doc:
            pdf_metadata = doc.metadata

            for page_number, page in enumerate(doc, start=1):
                for block in self._process_page(page):
                    text = block.text.strip()

                    if not text or text in seen:
                        continue

                    seen.add(text)
                    block.text = text
                    block.metadata.setdefault("page", page_number)
                    blocks.append(block)

        return DocumentRepresentation(
            author=pdf_metadata.get("author") or None,
            blocks=blocks,
        )
    
    def _process_page(self, page) -> list[DocumentBlock]:
        page_dict = page.get_text("dict")

        if self._text_length(page_dict) >= MIN_TEXT_THRESHOLD:
            return self._extract_text(page, page_dict)

        return self._extract_ocr(page, page_dict)
    
    @staticmethod
    def _text_length(page_dict) -> int:
        total = 0

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    total += len(span["text"])

                    if total >= MIN_TEXT_THRESHOLD:
                        return total
                    
        return total
    
    def _extract_text(self, page, page_dict) -> list[DocumentBlock]:
        tables = page.find_tables().tables
        table_rects = [pymupdf.Rect(table.bbox) for table in tables]

        output = []

        output.extend(self._extract_tables(tables))

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

            rect = pymupdf.Rect(block["bbox"])

            if any(rect.intersects(r) for r in table_rects):
                continue

            text = "".join(
                span["text"]
                for line in block["lines"]
                for span in line["spans"]
            ).strip()

            if text:
                output.append(
                    DocumentBlock(
                        type=DocumentBlockType.PARAGRAPH,
                        text=text,
                        metadata={},
                    )
                )

        return output
    
    def _extract_tables(self, tables) -> list[DocumentBlock]:
        output = []

        for table in tables:
            rows = table.extract()

            text = "\n".join(
                " | ".join((cell or "").strip() for cell in row)
                for row in rows
            ).strip()

            if text:
                output.append(
                    DocumentBlock(
                        type=DocumentBlockType.TABLE,
                        text=text,
                        metadata={},
                    )
                )

        return output
    
    def _extract_ocr(self, page, page_dict) -> list[DocumentBlock]:
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        largest = None
        largest_area = 0

        for block in page_dict["blocks"]:
            if block["type"] != 1:
                continue

            rect = pymupdf.Rect(block["bbox"])
            area = rect.width * rect.height

            if area > largest_area:
                largest = rect
                largest_area = area

        if largest and largest_area / page_area >= MIN_IMAGE_COVERAGE:
            pix = page.get_pixmap(
                clip=largest,
                dpi=OCR_DPI
            )
        else:
            pix = page.get_pixmap(
                dpi=OCR_DPI
            )

        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(
            mode,
            (pix.width, pix.height),
            pix.samples
        )
        
        text = self.ocr.extract_text(img)

        if not text:
            return []

        return [
            DocumentBlock(
                type=DocumentBlockType.OCR,
                text=text,
                metadata={},
            )
        ]