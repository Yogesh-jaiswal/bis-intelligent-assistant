import logging

from PIL import Image

# pytesseract is an optional runtime dependency.
# The app can start and serve requests without it; OCR is only needed
# when processing scanned-image PDFs.  Importing lazily prevents a
# ModuleNotFoundError at application startup / migration time.
try:
    import pytesseract as _pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _pytesseract = None
    _TESSERACT_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class OCR:
    def extract_text(self, img: Image.Image) -> str:
        """
        Extracts text from an image using Tesseract OCR.

        :param img: Pillow image object
        :return: Extracted text as a string
        """
        if not _TESSERACT_AVAILABLE:
            logger.warning(
                "pytesseract is not installed — OCR skipped. "
                "Install tesseract-ocr and pytesseract to enable OCR for scanned PDFs."
            )
            return ""
        try:
            text = _pytesseract.image_to_string(
                img,
                lang="eng+hin",
                config="--oem 3 psm 6"
            )
            return text.strip()
        except Exception:
            logger.exception("OCR failed")
            return ""