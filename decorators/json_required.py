import logging
from functools import wraps
from flask import request, g

from exceptions import BadRequestError

# Set up logging
logger = logging.getLogger(__name__)

def json_required(func):
    """Decorator to ensure that the request body contains valid JSON data."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent = True)

        if data is None:
            logger.warning("request rejected: body is not valid JSON")

            raise BadRequestError(
                "Request body must contain valid JSON"
            )
        
        g.json_data = data

        return func(*args, **kwargs)
    return wrapper