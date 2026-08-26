"""Chat and Conversation management endpoints."""

import uuid
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.core.deps import get_current_active_user, get_db
from backend.app.models.user import User
from backend.app.schemas.chat import (
    ChatRequest,
    ConversationCreate,
    ConversationListItem,
    ConversationResponse,
    MessageResponse,
)
from backend.app.services.chat_service import (
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversation_messages,
    get_user_conversations,
    stream_agent_chat,
)

router = APIRouter()


@router.get(
    "/conversations",
    response_model=List[ConversationListItem],
    summary="List all conversations for the current user",
)
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Returns conversation history threads for the authenticated user."""
    conversations = get_user_conversations(db, user_id=current_user.id)
    items = []
    for conv in conversations:
        items.append(
            ConversationListItem(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=len(conv.messages),
            )
        )
    return items


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation thread",
)
def create_new_conversation(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    conv_in: ConversationCreate,
) -> Any:
    """Creates a new empty conversation thread."""
    conv = create_conversation(
        db,
        user_id=current_user.id,
        title=conv_in.title or "New Conversation",
    )
    return conv


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get conversation details and full message history",
)
def get_conversation_details(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Fetches a specific conversation with all historical message turns."""
    conv = get_conversation(db, conversation_id=conversation_id, user_id=current_user.id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conv


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and all its messages",
)
def delete_user_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deletes a conversation thread."""
    success = delete_conversation(db, conversation_id=conversation_id, user_id=current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return None


@router.post(
    "/stream",
    summary="Stream agent response via Server-Sent Events (SSE)",
)
def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Executes the LangGraph agent and streams node events and final answer in real time."""
    return StreamingResponse(
        stream_agent_chat(db=db, user_id=current_user.id, request=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
