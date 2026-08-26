"""Chat service bridging LangGraph Agent execution, Supabase persistence, and SSE streaming."""

import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.models.chat import Conversation, Message
from backend.app.schemas.chat import ChatRequest
from src.graph import agent_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------
def create_conversation(
    db: Session,
    user_id: uuid.UUID,
    title: str = "New Conversation",
) -> Conversation:
    """Creates a new conversation thread for the given user."""
    conv = Conversation(
        user_id=user_id,
        title=title[:250],
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_user_conversations(
    db: Session,
    user_id: uuid.UUID,
    limit: int = 50,
) -> List[Conversation]:
    """Fetches all conversations for a user, ordered by most recently updated."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def get_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Optional[Conversation]:
    """Fetches a specific conversation belonging to a user."""
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )
    return db.scalars(stmt).first()


def delete_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    """Deletes a conversation and its cascaded messages."""
    conv = get_conversation(db, conversation_id=conversation_id, user_id=user_id)
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True


def get_conversation_messages(
    db: Session,
    conversation_id: uuid.UUID,
    limit: int = 200,
) -> List[Message]:
    """Fetches all messages in a conversation ordered chronologically."""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def add_message(
    db: Session,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    metadata_dict: Optional[dict] = None,
) -> Message:
    """Persists a message turn to the database and touches conversation updated_at."""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_json=metadata_dict or {},
    )
    db.add(msg)

    # Touch conversation updated_at timestamp
    conv = db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = func.now()

    db.commit()
    db.refresh(msg)
    return msg


def auto_generate_title(question: str) -> str:
    """Derives a concise conversation title from the first question."""
    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= 45:
        return cleaned
    return cleaned[:42] + "..."


# ---------------------------------------------------------------------------
# Agentic RAG Streaming Bridge
# ---------------------------------------------------------------------------
async def stream_agent_chat(
    db: Session,
    user_id: uuid.UUID,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Streams LangGraph execution updates and persists the interaction to Supabase."""

    # 1. Resolve or create conversation
    if request.conversation_id:
        conversation = get_conversation(
            db, conversation_id=request.conversation_id, user_id=user_id
        )
        if not conversation:
            error_data = json.dumps({"type": "error", "message": "Conversation not found"})
            yield f"data: {error_data}\n\n"
            return
    else:
        title = auto_generate_title(request.message)
        conversation = create_conversation(db, user_id=user_id, title=title)

    conv_id = conversation.id

    # 2. Persist user message to Supabase
    add_message(
        db,
        conversation_id=conv_id,
        role="user",
        content=request.message,
    )

    # 3. Load past message history for context
    db_messages = get_conversation_messages(db, conversation_id=conv_id)
    # Sliding window of last 10 turns (excluding the prompt we just added)
    history_window = db_messages[:-1][-10:]
    clean_history = [
        {"role": m.role, "content": m.content}
        for m in history_window
        if m.role in ("user", "assistant")
    ]

    # 4. Yield conversation initialization event
    init_event = json.dumps(
        {
            "type": "init",
            "conversation_id": str(conv_id),
            "title": conversation.title,
        }
    )
    yield f"data: {init_event}\n\n"

    # 5. Initialize Agent State
    state_input: dict = {
        "question": request.message,
        "chat_history": clean_history,
    }
    full_state: dict = dict(state_input)

    # 6. Stream graph node execution updates
    try:
        for chunk in agent_graph.stream(state_input, stream_mode="updates"):
            node_name, updates = next(iter(chunk.items()))

            if node_name.startswith("__"):
                continue

            if isinstance(updates, dict):
                full_state.update(updates)

            node_event = {
                "type": "node_update",
                "node": node_name,
                "data": {
                    "route": full_state.get("route"),
                    "thought": full_state.get("thought"),
                    "is_sufficient": full_state.get("is_sufficient"),
                    "retry_count": full_state.get("retry_count", 0),
                    "sources_count": len(full_state.get("sources", []) or []),
                },
            }
            yield f"data: {json.dumps(node_event)}\n\n"

        # 7. Extract final synthesized answer and metadata
        final_answer = full_state.get("answer") or full_state.get("direct_response") or "I was unable to synthesize an answer."
        route = full_state.get("route") or "unknown"
        sources = full_state.get("sources") or []
        thought = full_state.get("thought") or ""
        is_sufficient = full_state.get("is_sufficient")
        retry_count = full_state.get("retry_count", 0)

        metadata = {
            "route": route,
            "sources": sources,
            "thought": thought,
            "is_sufficient": is_sufficient,
            "retry_count": retry_count,
        }

        # 8. Persist assistant message to Supabase
        add_message(
            db,
            conversation_id=conv_id,
            role="assistant",
            content=final_answer,
            metadata_dict=metadata,
        )

        # 9. Yield final answer event
        final_event = json.dumps(
            {
                "type": "final_answer",
                "content": final_answer,
                "metadata": metadata,
            }
        )
        yield f"data: {final_event}\n\n"

    except Exception as e:
        logger.error(f"Error during agent graph streaming: {e}", exc_info=True)
        error_event = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_event}\n\n"

    # End of stream marker
    yield "data: [DONE]\n\n"
