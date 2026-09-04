from flask import Flask

from configs import get_settings
from configs.logging_config import configure_logging
from .extensions import db, migrate, limiter
from .commands.seed_dataset import seed_dataset_command
from routes.v1 import v1_bp
from middlewares.request_middlewares import register_middleware
from handlers.error_handlers import register_error_handlers

def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    if testing:
        app.config["TESTING"] = True

    # ------------------------------------------------------------------
    # Database configuration
    # ------------------------------------------------------------------
    settings = get_settings()
    app.config["SQLALCHEMY_DATABASE_URI"] = settings.DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = settings.SQLALCHEMY_TRACK_MODIFICATIONS

    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------
    db.init_app(app)
    migrate.init_app(app, db)
    limiter.init_app(app)

    # ------------------------------------------------------------------
    # Logging Configuration
    # ------------------------------------------------------------------
    configure_logging()

    # ------------------------------------------------------------------
    # Register Blueprints, Middlewares, and Global error handlers
    # ------------------------------------------------------------------
    app.register_blueprint(v1_bp)
    register_middleware(app)
    register_error_handlers(app)

    # ------------------------------------------------------------------
    # CLI commands
    # ------------------------------------------------------------------
    app.cli.add_command(seed_dataset_command)

    # ------------------------------------------------------------------
    # Preload Embedding Model at Application Startup
    # ------------------------------------------------------------------
    if settings.AI_PROVIDER != "FAKE" and not app.config.get("TESTING"):
        try:
            from services.file_processors.embeddings.embeddings_generator import init_embedding_model
            init_embedding_model()
        except Exception as e:
            app.logger.warning("[STARTUP] Non-fatal: embedding model preloading deferred: %s", e)

    return app