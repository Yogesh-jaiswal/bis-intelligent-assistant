from enum import Enum

class DocumentTypes(str, Enum):
    PDF = "pdf"

class DocumentBlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    OCR = "ocr"
    CODE = "code"
    LIST = "list"
    TRANSCRIPT = "transcript"
    DESCRIPTION = "description"