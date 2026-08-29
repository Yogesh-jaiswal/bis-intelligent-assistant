from flask import Flask, jsonify

from .extensions import db
from routes.v1 import v1_bp

def create_app() -> Flask:
    # app
    app = Flask(__name__)

    # Register Blueprints
    app.register_blueprint(v1_bp)

    # Register DB
    db.init_app(app)

    return app