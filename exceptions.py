# Unsupported Document Error
class UnsupportedDocumentError(Exception):
    pass

# Database Error
class DatabaseError(Exception):
    pass

# Bad Request Error
class BadRequestError(Exception):
    pass

# Resource Not Found Error
class ResourceNotFoundError(Exception):
    pass


# ============================================================
# AI PROVIDER EXCEPTIONS
# ============================================================

class AIProviderError(Exception):
    """Base exception for all AI provider and model errors."""
    def __init__(self, message: str, provider: str | None = None, model: str | None = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class AIConnectionError(AIProviderError):
    """Transport or connectivity errors (connection refused, timeout, DNS, tunnel down, HTTP failure)."""
    pass


class AIResponseError(AIProviderError):
    """Model response errors (empty response, missing message content, or malformed JSON)."""
    pass


class AISchemaValidationError(AIProviderError):
    """Structured output schema validation failure (Pydantic validation error or schema mismatch)."""
    pass


class AIServiceUnavailableError(AIConnectionError):
    """Alias for backwards compatibility mapping to 503."""
    pass