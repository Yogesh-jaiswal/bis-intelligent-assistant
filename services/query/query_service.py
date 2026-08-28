from typing import Any

from pydantic import BaseModel
from configs import get_settings

from services.ai.engine import AIEngine

class DummyResponse(BaseModel):
    response: str


def process_query(
    query: str,
) -> dict[str, Any]:
    """
    Process a user query using the configured AI provider.

    BIS retrieval, conversation context, citations, and
    structured data cards will be added later.
    """

    ai_engine = AIEngine(get_settings().AI_MODEL)

    return ai_engine.complete(
        prompt=query,
        response_schema=DummyResponse
    )