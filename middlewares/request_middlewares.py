import time
import uuid
import logging

from flask import Flask, g, request

# Set up logging
logger = logging.getLogger(__name__)

# Middleware to log request details and measure processing time
def register_middleware(app: Flask):
    """Registers middleware functions for logging and request tracking."""

    @app.before_request
    def before():
        """Executed before each request to set up request tracking."""
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.time()

        logger.info(f"[{g.request_id}] {request.method} {request.path} started")

    @app.after_request
    def after(response):
        """Executed after each request to log completion and duration."""
        request_id = getattr(g, "request_id", "unknown")
        start_time = getattr(g, "start_time", None)

        duration = 0

        if start_time is not None:
            duration = round(time.time() - start_time, 4)

        logger.info(
            f"[{g.request_id}] {request.method} {request.path} completed "
            f"in {duration}s with status {response.status_code}"
        )

        response.headers["X-Request-ID"] = request_id

        return response
    
    @app.teardown_request
    def teardown_request(error):
        """Executed after each request, regardless of exceptions, to log any teardown errors."""
        if error:
            request_id = getattr(g, "request_id", "unknown")

            logger.error(f"[{request_id}] teardown error: {error}")