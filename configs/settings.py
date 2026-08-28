from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    """Application settings for the BIS Intelligent Assistant."""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=5000)

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ] = Field(default="DEBUG")

    ENVIRONMENT: Literal[
        "development",
        "testing",
        "production",
        "evaluation",
    ] = Field(default="development")

    # ------------------------------------------------------------------
    # AI Settings
    # ------------------------------------------------------------------

    AI_MODEL: Literal["FAKE", "OLLAMA"] = Field(default="OLLAMA")

    # Ollama server URL.
    #
    # Local development:
    #   http://localhost:11434
    #
    # Google Colab + Cloudflare Tunnel:
    #   https://<your-tunnel>.trycloudflare.com
    #
    MODEL_URL: str = Field(
        default="http://localhost:11434"
    )

    MODEL_NAME: str = Field(
        default="your-ollama-model"
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    EMBEDDINGS_MODEL: str = Field(
        default="all-MiniLM-L6-v2"
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="bis_assistant")

    SQLALCHEMY_TRACK_MODIFICATIONS: bool = Field(
        default=False
    )

    HNSW_EF_SEARCH: int = Field(
        default=100,
        gt=0,
    )

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            "postgresql+psycopg://"
            f"{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    # ------------------------------------------------------------------
    # Database Query
    # ------------------------------------------------------------------

    MIN_SIMILARITY: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    MAX_LIMIT: int = Field(
        default=100,
        gt=0,
    )

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------

    MAX_CONTENT_LENGTH: int = Field(
        default=10485760,  # 10 MB
        gt=0,
    )

    UPLOAD_FOLDER: str = Field(
        default="file_uploads"
    )

    # ------------------------------------------------------------------
    # Settings configuration
    # ------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
    )
