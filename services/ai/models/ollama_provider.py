import json
import logging
import time
from typing import Any
import httpx
from ollama import Client, RequestError, ResponseError
from pydantic import BaseModel, ValidationError

from configs import get_settings
from exceptions import (
    AIConnectionError,
    AIResponseError,
    AISchemaValidationError,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaProvider:
    """Ollama provider for schema-enforced structured generation with granular error classification."""

    def __init__(self):
        self.client = Client(host=settings.MODEL_URL)
        self.model = settings.MODEL_NAME

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        think: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        schema_name = getattr(response_schema, "__name__", str(response_schema))
        logger.info(
            "[OLLAMA: REQUEST] Calling Ollama model='%s' at endpoint='%s' (schema='%s', think=%s)...",
            self.model,
            settings.MODEL_URL,
            schema_name,
            think,
        )

        chat_options: dict[str, Any] = {
            "temperature": 0.2,
            "num_predict": 500,
        }
        if options:
            chat_options.update(options)

        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": response_schema.model_json_schema(),
            "options": chat_options,
        }
        if think is not None:
            chat_kwargs["think"] = think

        start_t = time.perf_counter()

        # Step 1: Transport / HTTP invocation
        try:
            response = self.client.chat(**chat_kwargs)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, RequestError) as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(
                "[OLLAMA: CONNECTION FAILURE] Failed to connect to Ollama (model='%s', url='%s') after %.2f ms: %s",
                self.model,
                settings.MODEL_URL,
                elapsed_ms,
                exc,
            )
            raise AIConnectionError(
                f"Connection to Ollama server at '{settings.MODEL_URL}' failed: {exc}",
                provider="OLLAMA",
                model=self.model,
            ) from exc
        except ResponseError as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            status = getattr(exc, "status_code", "UNKNOWN")
            logger.warning(
                "[OLLAMA: HTTP FAILURE] Ollama server returned HTTP status %s (model='%s') after %.2f ms: %s",
                status,
                self.model,
                elapsed_ms,
                exc,
            )
            raise AIConnectionError(
                f"Ollama server returned HTTP {status}: {exc}",
                provider="OLLAMA",
                model=self.model,
            ) from exc
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.error(
                "[OLLAMA: UNEXPECTED TRANSPORT ERROR] Unexpected transport failure communicating with Ollama after %.2f ms: %s",
                elapsed_ms,
                exc,
            )
            raise AIConnectionError(
                f"Transport failure communicating with Ollama: {exc}",
                provider="OLLAMA",
                model=self.model,
            ) from exc

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Step 2: Content Extraction & Empty check
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if not message:
            logger.warning(
                "[OLLAMA: EMPTY RESPONSE] Ollama response missing 'message' field (model='%s') after %.2f ms",
                self.model,
                elapsed_ms,
            )
            raise AIResponseError(
                f"Ollama response missing message field from model '{self.model}'",
                provider="OLLAMA",
                model=self.model,
            )

        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if content is None or not str(content).strip():
            logger.warning(
                "[OLLAMA: EMPTY RESPONSE] Model '%s' returned empty content after %.2f ms",
                self.model,
                elapsed_ms,
            )
            raise AIResponseError(
                f"Model '{self.model}' returned an empty response content",
                provider="OLLAMA",
                model=self.model,
            )

        # Step 3: JSON parsing check
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[OLLAMA: MALFORMED OUTPUT] Model '%s' returned malformed JSON after %.2f ms: %s (content preview: '%s')",
                self.model,
                elapsed_ms,
                exc,
                str(content)[:100],
            )
            raise AIResponseError(
                f"Model '{self.model}' produced malformed JSON: {exc}",
                provider="OLLAMA",
                model=self.model,
            ) from exc

        # Step 4: Schema validation check
        try:
            validated = response_schema.model_validate(parsed_json)
            logger.info(
                "[OLLAMA: SUCCESS] Successfully parsed and validated '%s' from model '%s' in %.2f ms",
                schema_name,
                self.model,
                elapsed_ms,
            )
            return validated.model_dump()
        except ValidationError as exc:
            logger.warning(
                "[OLLAMA: SCHEMA VALIDATION FAILURE] Model '%s' output failed schema validation for '%s' after %.2f ms: %s",
                self.model,
                schema_name,
                elapsed_ms,
                exc,
            )
            raise AISchemaValidationError(
                f"Model '{self.model}' output does not conform to schema '{schema_name}': {exc}",
                provider="OLLAMA",
                model=self.model,
            ) from exc