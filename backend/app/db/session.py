"""Database session and engine management."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# NullPool is recommended when connecting through Supabase/Neon connection poolers
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=NullPool,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy database session per request."""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error(f"Database session rollback due to error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()
