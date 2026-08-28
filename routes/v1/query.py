from flask import request

from . import v1_bp
from validators.chat_responses import ChatRequest

from services.query.query_service import process_query
from utils.response_envelopes import create_success_response

@v1_bp.post("/query")
def query_endpoint():
    payload = ChatRequest(**request.json())

    result = process_query(payload.message)

    return create_success_response(result)