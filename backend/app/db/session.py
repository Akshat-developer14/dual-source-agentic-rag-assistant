"""Database session and connection engine management."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy Engine Initialization
# Configured with NullPool to allow Supabase/Neon connection poolers to manage pooling
# without client-side socket collisions, and pool_pre_ping for liveness checks.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Session Dependency Provider
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """FastAPI request-scoped dependency yielding a database session with auto-rollback on error."""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error("Database session encountered an error; executing rollback: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()
