import logging
from typing import Any

from flask import Flask, Response, jsonify
from flask_limiter.errors import RateLimitExceeded
from pydantic import ValidationError

from exceptions import (
    AIConnectionError,
    AIProviderError,
    AIResponseError,
    AISchemaValidationError,
    AIServiceUnavailableError,
    BadRequestError,
    DatabaseError,
    ResourceNotFoundError,
)
from utils.response_envelopes import create_error_response

logger = logging.getLogger(__name__)

# ============================================================
# VALIDATION ERROR HELPERS
# ============================================================

def reconstruct_validation_errors(
    errors: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Convert Pydantic validation errors into the API FieldError format.

    Example:

    Pydantic:
        {
            "loc": ("message", "content"),
            "msg": "Field required."
        }

    Becomes:
        {
            "field": "message.content",
            "message": "Field required."
        }
    """

    field_errors: list[dict[str, str]] = []

    for error in errors:
        location = error.get("loc", ())

        field = ".".join(str(part) for part in location)

        if not field:
            field = "request"

        field_errors.append(
            {
                "field": field,
                "message": error.get(
                    "msg",
                    "Invalid value.",
                ),
            }
        )

    return field_errors


# ============================================================
# ERROR HANDLERS
# ============================================================

def register_error_handlers(app: Flask) -> None:
    """Register all application-level error handlers."""

    # --------------------------------------------------------
    # Pydantic validation errors
    # --------------------------------------------------------

    @app.errorhandler(ValidationError)
    def handle_validation_errors(
        e: ValidationError,
    ) -> tuple[Response, int]:
        """Handle Pydantic request validation errors."""

        fields = reconstruct_validation_errors(e.errors())

        return create_error_response(
            error_type="VALIDATION_ERROR",
            message="Request validation failed.",
            status_code=422,
            fields=fields,
        )

    # --------------------------------------------------------
    # Bad request
    # --------------------------------------------------------

    @app.errorhandler(BadRequestError)
    def handle_bad_request(
        e: BadRequestError,
    ) -> tuple[Response, int]:
        """Handle invalid client requests."""

        return create_error_response(
            error_type="BAD_REQUEST_ERROR",
            message=str(e),
            status_code=400,
        )

    # --------------------------------------------------------
    # Resource not found
    # --------------------------------------------------------

    @app.errorhandler(ResourceNotFoundError)
    def handle_resource_not_found(
        e: ResourceNotFoundError,
    ) -> tuple[Response, int]:
        """Handle missing resources."""

        logger.warning("Resource not found: %s", e)

        return create_error_response(
            error_type="RESOURCE_NOT_FOUND",
            message="Requested resource not found.",
            status_code=404,
        )

    # --------------------------------------------------------
    # Rate limiting
    # --------------------------------------------------------

    @app.errorhandler(RateLimitExceeded)
    def handle_rate_limit(
        e: RateLimitExceeded,
    ) -> tuple[Response, int]:
        """Handle rate-limit violations."""

        logger.warning(
            "Rate limit exceeded: %s",
            e.description,
        )

        return create_error_response(
            error_type="RATE_LIMIT_ERROR",
            message="Too many requests. Please try again later.",
            status_code=429,
        )

    # --------------------------------------------------------
    # Database errors
    # --------------------------------------------------------

    @app.errorhandler(DatabaseError)
    def handle_database_error(
        e: DatabaseError,
    ) -> tuple[Response, int]:
        """Handle application database errors."""

        logger.exception(
            "Database error: %s",
            e,
        )

        return create_error_response(
            error_type="DATABASE_ERROR",
            message="A database error occurred.",
            status_code=500,
        )

    # --------------------------------------------------------
    # AI Service Errors (503 and 502)
    # --------------------------------------------------------

    @app.errorhandler(AIConnectionError)
    def handle_ai_connection_error(
        e: AIConnectionError,
    ) -> tuple[Response, int]:
        """Handle AI provider network/transport outage when no fallback is possible."""
        logger.warning("AI provider connection error (provider='%s', model='%s'): %s", e.provider, e.model, e)
        return create_error_response(
            error_type="AI_SERVICE_UNAVAILABLE",
            message="The AI service is currently unreachable. Please try again shortly.",
            status_code=503,
        )

    @app.errorhandler(AIResponseError)
    def handle_ai_response_error(
        e: AIResponseError,
    ) -> tuple[Response, int]:
        """Handle AI model invalid/empty/malformed response."""
        logger.warning("AI model response error (provider='%s', model='%s'): %s", e.provider, e.model, e)
        return create_error_response(
            error_type="AI_RESPONSE_ERROR",
            message="The AI service returned an unreadable response. Please rephrase your query or try again.",
            status_code=502,
        )

    @app.errorhandler(AISchemaValidationError)
    def handle_ai_schema_validation_error(
        e: AISchemaValidationError,
    ) -> tuple[Response, int]:
        """Handle AI structured output schema validation failure."""
        logger.warning("AI schema validation error (provider='%s', model='%s'): %s", e.provider, e.model, e)
        return create_error_response(
            error_type="AI_SCHEMA_VALIDATION_ERROR",
            message="The AI service produced an unexpected structured format.",
            status_code=502,
        )

    @app.errorhandler(AIProviderError)
    def handle_ai_provider_error(
        e: AIProviderError,
    ) -> tuple[Response, int]:
        """Catch-all for any other AI provider failure."""
        logger.warning("AI provider error (provider='%s', model='%s'): %s", e.provider, e.model, e)
        return create_error_response(
            error_type="AI_PROVIDER_ERROR",
            message="An error occurred while communicating with the AI service.",
            status_code=502,
        )

    # --------------------------------------------------------
    # Unexpected errors
    # --------------------------------------------------------

    @app.errorhandler(Exception)
    def handle_unexpected_error(
        e: Exception,
    ) -> tuple[Response, int]:
        """
        Catch unexpected errors so that internal exceptions never
        leak implementation details to the client.
        """

        logger.exception(
            "Unexpected server error: %s",
            e,
        )

        return create_error_response(
            error_type="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred.",
            status_code=500,
        )