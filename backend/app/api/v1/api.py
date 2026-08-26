"""Central API v1 Router aggregator."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import auth, chat

api_router = APIRouter()

# Include Auth endpoints under /auth
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Include Chat endpoints under /chat (to be populated)
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
