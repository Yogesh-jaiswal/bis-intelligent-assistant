import logging
import os

from flask import has_request_context, g
from logging.handlers import RotatingFileHandler

from configs import get_settings


class RequestIDFilter(logging.Filter):
    """Add request_id to log records."""

    def filter(self, record):
        if has_request_context():
            record.request_id = getattr(
                g,
                "request_id",
                "unknown"
            )
        else:
            record.request_id = "no-request"

        return True


def configure_logging():
    """Configure application logging."""

    settings = get_settings()

    root_logger = logging.getLogger()

    # Prevent duplicate handlers when create_app()
    # is called multiple times.
    if getattr(root_logger, "_configured", False):
        return
    
    root_logger._configured = True

    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | "
        "[%(request_id)s] | %(message)s"
    )

    request_filter = RequestIDFilter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)

    # Application log file
    app_file_handler = RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    app_file_handler.setFormatter(formatter)
    app_file_handler.addFilter(request_filter)

    # Error log file
    error_file_handler = RotatingFileHandler(
        filename="logs/error.log",
        maxBytes=1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    error_file_handler.addFilter(request_filter)

    root_logger.setLevel(
        getattr(
            logging,
            settings.LOG_LEVEL
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(error_file_handler)

    # Prevent Werkzeug from being overly noisy
    logging.getLogger("werkzeug").setLevel(logging.INFO)