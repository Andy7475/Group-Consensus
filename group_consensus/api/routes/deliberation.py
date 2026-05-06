"""Deliberation (Phase 1) endpoints — mediation loop."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from group_consensus.api.routes.sessions import _sessions
from group_consensus.models.types import Opinion

router = APIRouter(prefix="/sessions/{session_id}/deliberation", tags=["deliberation"])


class SubmitOpinionsRequest(BaseModel):
    opinions: list[dict]  # [{participant_id, text}]


class SubmitOpinionsResponse(BaseModel):
    session_id: str
    rounds_completed: int
    consensus_statement: str
    consensus_statement_id: str


@router.post("/run", response_model=SubmitOpinionsResponse)
def run_deliberation(
    session_id: str, req: SubmitOpinionsRequest
) -> SubmitOpinionsResponse:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    opinions = [
        Opinion(
            participant_id=op["participant_id"],
            text=op["text"],
            session_id=session_id,
        )
        for op in req.opinions
    ]

    result = session.run_mediation(opinions)

    return SubmitOpinionsResponse(
        session_id=session_id,
        rounds_completed=result.num_rounds,
        consensus_statement=result.consensus_statement.text,
        consensus_statement_id=str(result.consensus_statement.id),
    )


@router.get("/statements")
def list_statements(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "statements": [
            {"id": str(s.id), "text": s.text, "type": s.type, "round": s.round_number}
            for s in session._statements
        ],
    }
