import logging

from PIL import Image
import pytesseract

# Set up logging
logger = logging.getLogger(__name__)

class OCR:
    def extract_text(self, img: Image.Image) -> str:
        """
        Extracts text from an image using Tesseract OCR.

        :param img: Pillow image object
        :return: Extracted text as a string
        """
        try:
            text = pytesseract.image_to_string(
                img,
                lang = "eng+hin",
                config = "--oem 3 psm 6"
            )
            return text.strip()
        except Exception:
            logger.exception(f"OCR failed")
            return ""