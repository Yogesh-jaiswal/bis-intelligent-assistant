from . import v1_bp

from utils.response_envelopes import create_success_response

# Health Endpoint
@v1_bp.get("/health")
def health_endpoint():
    return create_success_response("BIS Intelligent agent is working")