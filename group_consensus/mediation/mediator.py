"""Orchestrates the full Habermas Machine deliberation loop.

Each round:
  1. StatementModel generates N candidate consensus statements
  2. RewardModel predicts how each participant ranks the candidates
  3. Schulze social-choice voting selects the winning statement
  4. Participants (or the facilitator) can submit critiques
  5. Critiques feed into the next round

The loop stops when max_rounds is reached or all participants accept the statement.
"""

from __future__ import annotations

from group_consensus.mediation.reward_model import RewardModel
from group_consensus.mediation.social_choice import select_winner_from_rankings
from group_consensus.mediation.statement_model import StatementModel
from group_consensus.models.types import (
    DeliberationRound,
    Opinion,
    Participant,
    SessionConfig,
    Statement,
    StatementType,
)


class MediationResult:
    def __init__(
        self,
        consensus_statement: Statement,
        rounds: list[DeliberationRound],
        session_id: str,
        topic: str,
    ) -> None:
        self.consensus_statement = consensus_statement
        self.rounds = rounds
        self.session_id = session_id
        self.topic = topic

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)


class Mediator:
    """Runs the iterative deliberation loop."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._statement_model = StatementModel(config)
        self._reward_model = RewardModel(config)

    def run(
        self,
        topic: str,
        participants: list[Participant],
        opinions: list[Opinion],
        critique_fn: "CritiqueFn | None" = None,
    ) -> MediationResult:
        """
        Run the deliberation loop synchronously.

        Args:
            topic: The question or issue being deliberated.
            participants: List of participants.
            opinions: Initial opinions, one per participant.
            critique_fn: Optional callable that takes the winning statement and
                         participants and returns a list of critique Opinions.
                         If None, the loop stops after one round.

        Returns:
            MediationResult with the final consensus statement and round history.
        """
        rounds: list[DeliberationRound] = []
        previous_winner: Statement | None = None
        critiques: list[Opinion] = []
        current_opinions = list(opinions)

        for round_num in range(self.config.max_deliberation_rounds):
            candidates = self._statement_model.generate_candidates(
                topic=topic,
                opinions=current_opinions,
                participants=participants,
                critiques=critiques if critiques else None,
                previous_winner=previous_winner,
                round_number=round_num,
            )

            rankings = self._reward_model.predict_rankings(
                participants=participants,
                opinions=current_opinions,
                candidates=candidates,
                session_id=self.config.session_id,
                round_number=round_num,
            )

            winner_id = select_winner_from_rankings(
                statement_ids=[s.id for s in candidates],
                participant_rankings=[r.statement_ids for r in rankings],
            )
            winner = next(s for s in candidates if s.id == winner_id)
            winner.type = StatementType.CONSENSUS

            new_critiques: list[Opinion] = []
            if critique_fn is not None:
                new_critiques = critique_fn(winner, participants)

            round_result = DeliberationRound(
                round_number=round_num,
                candidate_statements=candidates,
                rankings=rankings,
                winning_statement=winner,
                critiques=new_critiques,
            )
            rounds.append(round_result)

            previous_winner = winner
            critiques = new_critiques

            # Stop early if no critiques (group accepted the statement)
            if not new_critiques:
                break

        final_statement = rounds[-1].winning_statement
        final_statement.type = StatementType.REFINED

        return MediationResult(
            consensus_statement=final_statement,
            rounds=rounds,
            session_id=self.config.session_id,
            topic=topic,
        )


# Type alias for the critique callback
CritiqueFn = "Callable[[Statement, list[Participant]], list[Opinion]]"
