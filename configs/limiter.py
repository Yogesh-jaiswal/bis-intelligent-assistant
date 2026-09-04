from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from configs import get_settings

settings = get_settings()

# Initialize the Limiter with the appropriate configuration
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.RATELIMIT_ENABLED,
    default_limits=settings.DEFAULT_GLOBAL_LIMIT,
    storage_uri=settings.LIMITER_STORAGE_URI,
    strategy=settings.LIMITER_STRATEGY,
    headers_enabled=settings.RATELIMIT_HEADERS_ENABLED
)