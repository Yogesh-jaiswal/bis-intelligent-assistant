"""
Document extraction services.

Provides processors for extracting usable content from
supported file and external content sources.
"""

import time
from models.enums import DocumentTypes

from .pdf_processor import PDFProcessor
from .html_processor import HTMLProcessor

from exceptions import UnsupportedDocumentError

class DocumentProcessorFactory:
    """A factory class to create different types of document processors based on the specified file type."""
    PROCESSORS = {
        DocumentTypes.PDF: PDFProcessor,
        DocumentTypes.HTML: HTMLProcessor,
    }

    @staticmethod
    def get_processor(file_type: str, test_mode: bool = False):
        """Get the file processor according to file extension"""
        if test_mode:
            time.sleep(0) # Simulate a delay for testing purposes

        processor_class = DocumentProcessorFactory.PROCESSORS.get(file_type)

        if not processor_class:
            raise UnsupportedDocumentError(f"Unsupported file type {file_type}")
        
        return processor_class()