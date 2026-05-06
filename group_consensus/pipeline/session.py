"""End-to-end consensus session combining mediation and clustering.

A ConsensusSession runs through both phases:
  Phase 1: Habermas Machine deliberation to produce refined statements
  Phase 2: Polis-style wide voting and clustering to find group consensus

The session holds all state and can be used interactively (votes added
incrementally) or in batch mode for testing and demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from group_consensus.clustering.consensus import ConsensusDetector
from group_consensus.clustering.engine import ClusteringEngine
from group_consensus.clustering.opinion_matrix import OpinionMatrix
from group_consensus.mediation.mediator import MediationResult, Mediator
from group_consensus.models.types import (
    ConsensusResult,
    Opinion,
    Participant,
    SessionConfig,
    Statement,
    StatementType,
    Vote,
    VoteValue,
)


@dataclass
class SessionResult:
    mediation: MediationResult
    consensus: ConsensusResult
    session_id: str
    topic: str

    @property
    def summary(self) -> str:
        lines = [
            f"Topic: {self.topic}",
            f"Mediation rounds: {self.mediation.num_rounds}",
            f"Final consensus statement: {self.mediation.consensus_statement.text}",
            f"Opinion clusters found: {self.consensus.clustering.num_clusters}",
            f"Bridging statements: {len(self.consensus.bridging_statements)}",
            f"Divisive statements: {len(self.consensus.divisive_statements)}",
        ]
        if self.consensus.bridging_statements:
            lines.append("\nTop bridging statement:")
            lines.append(f"  {self.consensus.bridging_statements[0].text}")
        return "\n".join(lines)


class ConsensusSession:
    """
    Manages a full deliberation session from initial opinions to consensus map.

    Usage:
        session = ConsensusSession(config)
        result = session.run(participants, opinions, votes)

    Or incrementally:
        session = ConsensusSession(config)
        mediation_result = session.run_mediation(participants, opinions)
        session.add_vote(vote)
        consensus_result = session.run_clustering()
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._mediator = Mediator(config)
        self._opinion_matrix = OpinionMatrix()
        self._statements: list[Statement] = []
        self._participants: list[Participant] = []
        self._mediation_result: MediationResult | None = None

    def add_participant(self, participant: Participant) -> None:
        self._participants.append(participant)
        self._opinion_matrix.add_participant(participant.id)

    def add_vote(self, vote: Vote) -> None:
        self._opinion_matrix.add_vote(vote)

    def add_statement(self, statement: Statement) -> None:
        self._statements.append(statement)
        self._opinion_matrix.add_statement(statement)

    def run_mediation(
        self,
        opinions: list[Opinion],
        critique_fn=None,
    ) -> MediationResult:
        """
        Run Phase 1: deliberation loop.
        Participants must have been added via add_participant() first.
        """
        result = self._mediator.run(
            topic=self.config.topic,
            participants=self._participants,
            opinions=opinions,
            critique_fn=critique_fn,
        )
        self._mediation_result = result

        # Load the refined statement and all candidates into the opinion matrix
        for rnd in result.rounds:
            for stmt in rnd.candidate_statements:
                self.add_statement(stmt)
        self.add_statement(result.consensus_statement)

        return result

    def run_clustering(self) -> ConsensusResult:
        """
        Run Phase 2: cluster participants by their votes and detect consensus.
        Requires that votes have been added via add_vote().
        """
        if not self._statements:
            raise RuntimeError("No statements loaded. Run run_mediation() or add statements first.")

        return ConsensusDetector.from_votes(
            statements=self._statements,
            opinion_matrix=self._opinion_matrix,
            max_clusters=self.config.max_clusters,
            min_cluster_approval=self.config.min_cluster_approval,
        )

    def run(
        self,
        opinions: list[Opinion],
        votes: list[Vote],
        critique_fn=None,
    ) -> SessionResult:
        """
        Run the full pipeline: mediation then clustering.

        Args:
            opinions: One Opinion per participant for the initial deliberation.
            votes: Votes on statements (can include votes on mediation outputs).
            critique_fn: Optional callback for participant critiques between rounds.
        """
        mediation = self.run_mediation(opinions, critique_fn=critique_fn)

        for vote in votes:
            self.add_vote(vote)

        consensus = self.run_clustering()

        return SessionResult(
            mediation=mediation,
            consensus=consensus,
            session_id=self.config.session_id,
            topic=self.config.topic,
        )

    @classmethod
    def create(
        cls,
        topic: str,
        participants: list[dict],
        **kwargs,
    ) -> "ConsensusSession":
        """
        Convenience constructor.

        Args:
            topic: The question or issue being deliberated.
            participants: List of dicts with at least 'id' and 'name'.
            **kwargs: Overrides for SessionConfig fields.
        """
        config = SessionConfig(topic=topic, **kwargs)
        session = cls(config)
        for p in participants:
            session.add_participant(Participant(**p))
        return session
