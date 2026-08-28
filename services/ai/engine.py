from typing import Any

from pydantic import BaseModel

from .models.fake_provider import FakeProvider
from .models.ollama_provider import OllamaProvider

class AIEngine:
    """A class that provides an interface to different AI providers for generating content based on a given response schema."""
    AI_PROVIDERS = {
        "FAKE": FakeProvider,
        "OLLAMA": OllamaProvider
    }

    def __init__(self, provider):
        self.provider = provider

    def complete(self, prompt: str, response_schema: type[BaseModel]) -> dict[str, Any]:
        """Generate content using the specified AI provider."""
        model = self.AI_PROVIDERS.get(self.provider)

        if not model:
            raise ValueError("Invalid AI provider")
        
        return model().generate(prompt, response_schema)