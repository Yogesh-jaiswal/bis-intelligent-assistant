from typing import Any

from ollama import Client
from pydantic import BaseModel

from configs import get_settings

# settings
settings = get_settings()

class OllamaProvider:
    """Ollama provider for schema-enforced structured generation."""

    def __init__(self):
        self.client = Client(
            host=settings.MODEL_URL
        )

        self.model = settings.MODEL_NAME

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> dict[str, Any]:

        response = self.client.chat(
            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            format=response_schema.model_json_schema(),

            options={
                "temperature": 0.2,
                "num_predict": 500,
            },
        )

        content = response["message"]["content"]

        result = response_schema.model_validate_json(content)

        return result.model_dump()