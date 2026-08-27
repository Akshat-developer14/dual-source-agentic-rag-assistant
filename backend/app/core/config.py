"""Application configuration and environment settings management.

Loads and validates system-wide settings using Pydantic Settings,
including database URIs, JWT cryptographic parameters, CORS rules, and API keys.
"""

from typing import List, Optional, Union
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration schema for FastAPI, Database, Auth, and external AI services."""

    # ---------------------------------------------------------------------------
    # Application & API Metadata
    # ---------------------------------------------------------------------------
    PROJECT_NAME: str = "Kara Agentic RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # ---------------------------------------------------------------------------
    # External AI & Search Service Credentials
    # ---------------------------------------------------------------------------
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    # ---------------------------------------------------------------------------
    # Database Configuration (Supabase / PostgreSQL)
    # ---------------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
        description="Unified PostgreSQL database connection URI string",
    )

    # ---------------------------------------------------------------------------
    # JWT Authentication & Cryptographic Security
    # ---------------------------------------------------------------------------
    SECRET_KEY: str = Field(
        default="kara-default-insecure-secret-key-change-in-production",
        description="HMAC secret key used for signing and verifying JWT access tokens",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7-day token lifespan

    # ---------------------------------------------------------------------------
    # CORS Configuration
    # ---------------------------------------------------------------------------
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str]) -> str:
        """Normalizes database URI to enforce the modern psycopg driver and SSL requirements.
        
        Args:
            v: Raw connection string from environment variables.
            
        Returns:
            Formatted PostgreSQL connection string with appropriate driver prefix and query parameters.
        """
        if not v:
            return "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"

        url = v.strip().strip("'\"")
        # Ensure psycopg driver prefix is present for SQLAlchemy 2.0 compatibility
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)

        # Enforce SSL encryption for managed cloud providers (Supabase / Neon)
        if "sslmode=" not in url and ("supabase.com" in url or "neon.tech" in url):
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}sslmode=require"

        return url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
