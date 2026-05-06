"""Voting (Phase 2) endpoints — Polis-style statement voting."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from group_consensus.api.routes.sessions import _sessions
from group_consensus.models.types import Vote, VoteValue

router = APIRouter(prefix="/sessions/{session_id}/votes", tags=["voting"])


class CastVoteRequest(BaseModel):
    participant_id: str
    statement_id: str
    value: int  # +1, 0, -1


class CastVoteResponse(BaseModel):
    recorded: bool
    next_statements: list[str]


class ClusteringResponse(BaseModel):
    session_id: str
    num_clusters: int
    silhouette_score: float
    bridging_statements: list[dict]
    divisive_statements: list[dict]
    coordinates_2d: dict[str, list[float]]


@router.post("/", response_model=CastVoteResponse)
def cast_vote(session_id: str, req: CastVoteRequest) -> CastVoteResponse:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        value = VoteValue(req.value)
    except ValueError:
        raise HTTPException(status_code=400, detail="value must be +1, 0, or -1")

    vote = Vote(
        participant_id=req.participant_id,
        statement_id=UUID(req.statement_id),
        value=value,
        session_id=session_id,
    )
    session.add_vote(vote)

    next_stmts = session._opinion_matrix.next_statements_for_participant(
        req.participant_id, n=5
    )

    return CastVoteResponse(
        recorded=True,
        next_statements=[str(sid) for sid in next_stmts],
    )


@router.get("/cluster", response_model=ClusteringResponse)
def get_clustering(session_id: str) -> ClusteringResponse:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = session.run_clustering()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ClusteringResponse(
        session_id=session_id,
        num_clusters=result.clustering.num_clusters,
        silhouette_score=result.clustering.silhouette_score,
        bridging_statements=[
            {"id": str(s.id), "text": s.text} for s in result.bridging_statements
        ],
        divisive_statements=[
            {"id": str(s.id), "text": s.text} for s in result.divisive_statements
        ],
        coordinates_2d=result.clustering.coordinates_2d,
    )
