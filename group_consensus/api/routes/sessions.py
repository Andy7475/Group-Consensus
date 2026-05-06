"""Session management endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from group_consensus.models.types import Participant, SessionConfig
from group_consensus.pipeline.session import ConsensusSession

router = APIRouter(prefix="/sessions", tags=["sessions"])

# In-memory session store (replace with DB persistence)
_sessions: dict[str, ConsensusSession] = {}


class CreateSessionRequest(BaseModel):
    topic: str
    participants: list[dict]
    num_candidate_statements: int = 8
    max_deliberation_rounds: int = 5
    min_cluster_approval: float = 0.6


class CreateSessionResponse(BaseModel):
    session_id: str
    topic: str
    participant_count: int


@router.post("/", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    session = ConsensusSession.create(
        topic=req.topic,
        participants=req.participants,
        num_candidate_statements=req.num_candidate_statements,
        max_deliberation_rounds=req.max_deliberation_rounds,
        min_cluster_approval=req.min_cluster_approval,
    )
    _sessions[session.config.session_id] = session
    return CreateSessionResponse(
        session_id=session.config.session_id,
        topic=req.topic,
        participant_count=len(req.participants),
    )


@router.get("/{session_id}")
def get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "topic": session.config.topic,
        "participant_count": len(session._participants),
        "statement_count": len(session._statements),
        "vote_count": len(session._opinion_matrix),
    }
