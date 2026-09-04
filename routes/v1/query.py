import logging
import time
from flask import g

from . import v1_bp
from validators.chat_responses import ChatRequest
from services.query.query_service import process_query
from utils.response_envelopes import create_success_response
from app.extensions import limiter
from configs import get_settings
from decorators.json_required import json_required

logger = logging.getLogger(__name__)
settings = get_settings()


@v1_bp.post("/query")
@limiter.limit(settings.QUERY_RATE_LIMIT, override_defaults=False)
@json_required
def query_endpoint():
    start_time = time.perf_counter()
    conv_id = g.json_data.get("conversation_id", "unknown")
    msg_preview = str(g.json_data.get("message", {}).get("content", ""))[:80]
    
    logger.info(
        "[API: /v1/query] Incoming query request (conversation_id='%s', prompt_preview='%s')",
        conv_id,
        msg_preview,
    )
    
    try:
        payload = ChatRequest(**g.json_data)
    except Exception as e:
        logger.warning(
            "[API: /v1/query] Request validation failed (conversation_id='%s'): %s",
            conv_id,
            e,
        )
        raise

    try:
        result = process_query(payload)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        msg_type = result.get("message_type", "unknown")
        cards_count = len(result.get("data", []))
        cit_count = len(result.get("citations", []))
        
        logger.info(
            "[API: /v1/query] Request completed in %.2f ms (conversation_id='%s', type='%s', data_cards=%d, citations=%d)",
            elapsed_ms,
            conv_id,
            msg_type,
            cards_count,
            cit_count,
        )
        return create_success_response(result)
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            "[API: /v1/query] Pipeline error after %.2f ms (conversation_id='%s'): %s",
            elapsed_ms,
            conv_id,
            e,
            exc_info=True,
        )
        raise