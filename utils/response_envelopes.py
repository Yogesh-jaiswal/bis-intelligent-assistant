from typing import Any
from flask import Response, jsonify


def create_success_response(success_response: dict[str, Any] | str) -> dict[str, Any]:
    """Create a success response envelope."""
    return {
        "success": True,
        "data": success_response,
        "error": None,
    }


def create_error_envelope(error: dict[str, Any]) -> dict[str, Any]:
    """Wrap error dictionary in error envelope."""
    return {
        "success": False,
        "data": None,
        "error": error,
    }


def create_error_response(
    error_type: str,
    message: str,
    status_code: int,
    fields: list[dict[str, str]] | None = None,
) -> tuple[Response, int]:
    """
    Create a response matching the API ErrorResponse contract.

    Shape:
    {
        "success": false,
        "data": null,
        "error": {
            "type": "...",
            "message": "...",
            "status_code": 400,
            "fields": [...]
        }
    }
    """
    error: dict[str, Any] = {
        "type": error_type,
        "message": message,
        "status_code": status_code,
    }

    if fields is not None:
        error["fields"] = fields

    return jsonify(create_error_envelope(error)), status_code