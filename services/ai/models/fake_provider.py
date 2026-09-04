from typing import Any

from pydantic import BaseModel
from polyfactory.factories.pydantic_factory import ModelFactory


class FakeProvider:
    """A fake provider that generates fake data based on a given response schema."""
    
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        generation_options: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:

        generation_options = generation_options or {}
        overrides = generation_options.get("fake_provider", {})

        class Factory(ModelFactory):
            __model__ = response_schema

        obj = Factory.build()

        for field_name, length in overrides.items():

            if not hasattr(obj, field_name):
                continue

            value = getattr(obj, field_name)

            if not isinstance(value, list):
                continue

            annotation = response_schema.model_fields[field_name].annotation

            item_type = annotation.__args__[0]

            class ItemFactory(ModelFactory):
                __model__ = item_type

            setattr(
                obj,
                field_name,
                [ItemFactory.build() for _ in range(length)],
            )

        return obj.model_dump()