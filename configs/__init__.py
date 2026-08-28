from functools import cache
from .settings import BaseAppSettings

@cache
def get_settings():
    return BaseAppSettings()