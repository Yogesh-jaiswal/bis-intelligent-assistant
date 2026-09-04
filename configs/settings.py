from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, computed_field, field_validator
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

    AI_PROVIDER: str = Field(
        default="OLLAMA",
        validation_alias=AliasChoices("AI_PROVIDER", "AI_BACKEND")
    )

    # Ollama server URL.
    #
    # Local development:
    #   http://localhost:11434
    #
    # Google Colab + Cloudflare Tunnel:
    #   https://<your-tunnel>.trycloudflare.com
    #
    MODEL_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("MODEL_URL", "AI_BASE_URL")
    )

    MODEL_NAME: str = Field(
        default="qwen:8b",
        validation_alias=AliasChoices("MODEL_NAME", "AI_MODEL")
    )

    @field_validator("MODEL_URL", "MODEL_NAME", "AI_PROVIDER", mode="before")
    @classmethod
    def _strip_strings(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @property
    def AI_MODEL(self) -> str:
        """Normalized AI provider name ('FAKE' or 'OLLAMA') for AIEngine."""
        val = (self.AI_PROVIDER or "OLLAMA").strip().upper()
        return val if val in ("FAKE", "OLLAMA") else "OLLAMA"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    EMBEDDINGS_MODEL: str = Field(
        default="all-MiniLM-L6-v2"
    )
    EMBEDDINGS_DEVICE: str | None = Field(
        default=None,
        description="Device for embeddings ('cuda', 'cpu', or None for auto-detection)."
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    POSTGRES_HOST: str = Field(default="127.0.0.1")
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
    # Rate Limiting
    # ------------------------------------------------------------------

    RATELIMIT_ENABLED: bool = Field(default=True)
    DEFAULT_GLOBAL_LIMIT: list[str] = Field(default_factory=lambda: ["200 per day", "50 per hour"])
    LIMITER_STORAGE_URI: str = Field(default="memory://")
    LIMITER_STRATEGY: str = Field(default="fixed-window")
    RATELIMIT_HEADERS_ENABLED: bool = Field(default=True)

    QUERY_RATE_LIMIT: str = Field(default="15 per minute")


    # ------------------------------------------------------------------
    # Retrieval & Multi-Hop Controller Settings
    # ------------------------------------------------------------------

    MAX_RETRIEVAL_HOPS: int = Field(
        default=3,
        gt=0,
        description="Maximum sequential retrieval iterations permitted for multi-hop queries.",
    )

    DEFAULT_RAG_TOP_K: int = Field(
        default=5,
        gt=0,
        description="Default number of documentary passages retrieved per vector search.",
    )

    DEFAULT_DB_LIMIT: int = Field(
        default=10,
        gt=0,
        description="Default record limit per structured database operation.",
    )

    SEMANTIC_DB_SIMILARITY_THRESHOLD: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Cosine similarity cutoff for semantic database entity discovery.",
    )

    MAX_EVIDENCE_COUNT: int = Field(
        default=25,
        gt=0,
        description="Maximum accumulated evidence records/chunks before triggering final synthesis.",
    )

    CLARIFICATION_THRESHOLD: int = Field(
        default=3,
        gt=1,
        description="Threshold of ambiguous entity matches triggering evidence-based clarification.",
    )

    # ------------------------------------------------------------------
    # Settings configuration
    # ------------------------------------------------------------------


    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
    )
