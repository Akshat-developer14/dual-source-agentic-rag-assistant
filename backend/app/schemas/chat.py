"""Pydantic schemas for Conversation, Message, and Chat Streaming."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Message Schemas
# ---------------------------------------------------------------------------
class MessageBase(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_json",
        serialization_alias="metadata",
    )


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, description="Message text content")


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_json",
        serialization_alias="metadata",
    )
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Conversation Schemas
# ---------------------------------------------------------------------------
class ConversationBase(BaseModel):
    title: str = "New Conversation"


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationResponse(ConversationBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ConversationListItem(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Chat Execution / Streaming Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User prompt query")
    conversation_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Existing conversation ID. If None, a new conversation will be created.",
    )


class StreamChunk(BaseModel):
    type: Literal["node_update", "token", "final_answer", "error"]
    node: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
