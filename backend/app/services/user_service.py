"""User management and authentication CRUD operations."""

import uuid
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.security import get_password_hash, verify_password
from backend.app.models.user import User
from backend.app.schemas.user import UserCreate, UserUpdate


# ---------------------------------------------------------------------------
# User Query Functions
# ---------------------------------------------------------------------------
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieves a single user record matching the normalized email address."""
    stmt = select(User).where(User.email == email.strip().lower())
    return db.scalars(stmt).first()


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Retrieves a single user record by its primary UUID."""
    stmt = select(User).where(User.id == user_id)
    return db.scalars(stmt).first()


# ---------------------------------------------------------------------------
# User Lifecycle & Authentication Functions
# ---------------------------------------------------------------------------
def create_user(db: Session, user_in: UserCreate) -> User:
    """Persists a new user record with a securely hashed bcrypt password."""
    db_user = User(
        email=user_in.email.strip().lower(),
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name.strip() if user_in.full_name else None,
        is_active=True,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> Optional[User]:
    """Authenticates credentials against stored password hashes.
    
    Returns:
        User model instance if authentication succeeds, else None.
    """
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def update_user(
    db: Session,
    db_user: User,
    user_update: UserUpdate,
) -> User:
    """Updates user profile information or replaces password hash."""
    if user_update.full_name is not None:
        db_user.full_name = user_update.full_name
    if user_update.password is not None:
        db_user.hashed_password = get_password_hash(user_update.password)
    db.commit()
    db.refresh(db_user)
    return db_user
