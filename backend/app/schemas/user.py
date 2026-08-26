"""Pydantic schemas for User entities and Authentication."""

import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# Base user schema
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True


# Schema for user registration
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128, description="User cleartext password")
    full_name: Optional[str] = None


# Schema for user profile update
class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=6, max_length=128)


# Schema for user public response
class UserResponse(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Schema for JWT Authentication Token response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Schema for decoded JWT token payload
class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
