import logging
import time
from typing import Any
from pydantic import BaseModel

from .models.fake_provider import FakeProvider
from .models.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class AIEngine:
    """A class that provides an interface to different AI providers for generating content based on a given response schema."""
    AI_PROVIDERS = {
        "FAKE": FakeProvider,
        "OLLAMA": OllamaProvider,
    }

    def __init__(self, provider: str):
        self.provider = (provider or "FAKE").upper()

    def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        think: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate content using the specified AI provider."""
        model_cls = self.AI_PROVIDERS.get(self.provider)

        if not model_cls:
            logger.error("[AI ENGINE] Invalid AI provider '%s' requested", self.provider)
            raise ValueError(f"Invalid AI provider: '{self.provider}'")

        schema_name = getattr(response_schema, "__name__", str(response_schema))
        logger.info(
            "[AI ENGINE: REQUEST] Dispatching completion to provider '%s' (schema='%s', prompt_len=%d, think=%s)",
            self.provider,
            schema_name,
            len(prompt),
            think,
        )

        start_t = time.perf_counter()
        try:
            provider_inst = model_cls()
            if self.provider == "OLLAMA":
                result = provider_inst.generate(
                    prompt,
                    response_schema,
                    think=think,
                    options=options,
                )
            else:
                result = provider_inst.generate(prompt, response_schema)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(
                "[AI ENGINE: SUCCESS] Completed generation via '%s' (model='%s') in %.2f ms",
                self.provider,
                getattr(provider_inst, "model", self.provider),
                elapsed_ms,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(
                "[AI ENGINE: ERROR] Generation failed via '%s' after %.2f ms (%s): %s",
                self.provider,
                elapsed_ms,
                type(exc).__name__,
                exc,
            )
            raise