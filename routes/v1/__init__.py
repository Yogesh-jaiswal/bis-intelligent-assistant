from flask import Blueprint

v1_bp = Blueprint("v1", __name__, url_prefix="/v1")

from . import health
from . import query