"""Authentication and user session endpoints."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.core.deps import get_current_active_user, get_db
from backend.app.core.security import create_access_token
from backend.app.models.user import User
from backend.app.schemas.user import Token, UserCreate, UserResponse
from backend.app.services.user_service import authenticate_user, create_user, get_user_by_email

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """Creates a new user with email and password."""
    existing_user = get_user_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )
    user = create_user(db, user_in=user_in)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="Authenticate and obtain JWT access token (OAuth2 Form)",
)
def login_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """OAuth2 compatible token login, getting an access token for future requests."""
    user = authenticate_user(
        db,
        email=form_data.username,
        password=form_data.password,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account.",
        )

    access_token = create_access_token(subject=str(user.id))
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current logged-in user profile",
)
def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Fetches profile details of the authenticated user."""
    return current_user
