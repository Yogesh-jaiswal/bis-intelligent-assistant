from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from configs.limiter import limiter 

db = SQLAlchemy()
migrate = Migrate()

__all__ = [
    "limiter"
]