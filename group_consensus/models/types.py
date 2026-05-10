"""Core data models for the Group Consensus platform."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FileSerializable(BaseModel):
    """Mixin that adds to_file / from_file to any Pydantic model."""

    def to_file(self, path: Path | str) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        return cls.model_validate_json(Path(path).read_text())


class VoteValue(int, Enum):
    AGREE = 1
    PASS = 0
    DISAGREE = -1


class StatementType(str, Enum):
    ATOMIC = "atomic"          # short single-idea statement for Pol.is voting
    CONTESTED = "contested"    # atomic statement where opinions directly conflict
    CANDIDATE = "candidate"    # comprehensive candidate (legacy / final synthesis)
    CONSENSUS = "consensus"
    REFINED = "refined"
    PARTICIPANT_OPINION = "participant_opinion"


class Participant(BaseModel):
    id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Opinion(BaseModel):
    participant_id: str
    text: str
    session_id: str


class Statement(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    type: StatementType
    session_id: str
    round_number: int = 0
    source_participant_id: str | None = None

    def __hash__(self) -> int:
        return hash(self.id)


class Vote(BaseModel):
    participant_id: str
    statement_id: UUID
    value: VoteValue
    session_id: str


class PreferenceRanking(BaseModel):
    """Ordered list of statement IDs from most to least preferred."""

    participant_id: str
    statement_ids: list[UUID]
    session_id: str
    round_number: int


class DeliberationRound(BaseModel):
    round_number: int
    candidate_statements: list[Statement]
    rankings: list[PreferenceRanking]
    winning_statement: Statement
    critiques: list[Opinion] = Field(default_factory=list)


class Cluster(BaseModel):
    id: int
    participant_ids: list[str]
    centroid: list[float]
    label: str | None = None


class ClusteringResult(BaseModel):
    clusters: list[Cluster]
    coordinates_2d: dict[str, list[float]]
    num_clusters: int
    silhouette_score: float
    statement_approval_by_cluster: dict[str, dict[int, float]]


class ConsensusResult(FileSerializable):
    """Statements that achieve high approval across all opinion clusters."""

    bridging_statements: list[Statement]
    divisive_statements: list[Statement]
    clustering: ClusteringResult
    group_consensus_threshold: float


class MergeRecord(BaseModel):
    """Records what the compression step merged and why."""
    kept: str
    absorbed: list[str]
    reason: str


class CompressionResult(BaseModel):
    statements: list[Statement]
    merges: list[MergeRecord]
    contested_pairs: list[tuple[str, str]]   # text pairs in direct conflict


class SessionConfig(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    num_candidate_statements: int = 8
    max_deliberation_rounds: int = 5
    min_cluster_approval: float = 0.6
    max_clusters: int = 5
    claude_model: str = "claude-sonnet-4-6"
