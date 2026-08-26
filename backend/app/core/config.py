"""Application configuration settings using Pydantic Settings."""

from typing import List, Optional, Union
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for FastAPI, Database, Auth, and Services."""

    PROJECT_NAME: str = "Kara Agentic RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Core AI Service Keys
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    # Single Unified Database Connection URL
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
        description="Unified PostgreSQL database connection string",
    )

    # JWT Authentication & Security
    SECRET_KEY: str = Field(
        default="kara-default-insecure-secret-key-change-in-production",
        description="Secret key used for signing and verifying JWT tokens",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7-day token expiration

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str]) -> str:
        """Normalize database URI to use the psycopg driver and require SSL for cloud Postgres."""
        if not v:
            return "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"

        url = v.strip().strip("'\"")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)

        # Ensure sslmode=require is appended for Supabase/Neon if missing
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
