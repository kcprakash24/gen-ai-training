"""
Chat API endpoints with conversation memory management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid

from agents.supervisor import Supervisor
from core.telemetry import telemetry

router = APIRouter()

# In-memory conversation storage (replace with Redis/DB in production)
conversations: Dict[str, List[Dict]] = {}

# Initialize supervisor agent
supervisor = Supervisor()  # ✅ Changed from SupervisorAgent()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    metadata: Optional[dict] = None


@router.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message with conversation memory."""

    # Create or retrieve session
    session_id = request.session_id or str(uuid.uuid4())

    # Initialize conversation if new session
    if session_id not in conversations:
        conversations[session_id] = []
        # Start MLflow tracking for this session
        telemetry.start_session(session_id)

    # Add user message to history
    conversations[session_id].append(
        {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
        }
    )

    print(f"Received chat request: {request.message} (session: {session_id})")

    # Get conversation context (last 5 messages)
    context = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversations[session_id][-5:]]
    )

    # Process with supervisor
    response = supervisor.process(
        query=request.message, context=context, session_id=session_id
    )

    # Add assistant response to history
    conversations[session_id].append(
        {
            "role": "assistant",
            "content": response.answer,
            "timestamp": datetime.now().isoformat(),
        }
    )

    return ChatResponse(
        answer=response.answer, session_id=session_id, metadata=response.metadata
    )


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "messages": conversations[session_id]}


@router.delete("/chat/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversations:
        # End MLflow tracking for this session
        telemetry.end_session()
        del conversations[session_id]

    return {"message": "History cleared", "session_id": session_id}
