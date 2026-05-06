"""LLM-as-judge reward model for predicting participant preference rankings.

The original Habermas Machine used a fine-tuned reward model trained on human
preference data. This implementation uses Claude as a zero-shot judge: for each
participant, it considers their opinion and ranks the candidate statements by
how well each one represents that participant's perspective.

This is a practical approximation for deployment without training data.
As preference data accumulates, this module can be replaced with a trained model.
"""

from __future__ import annotations

from uuid import UUID

import anthropic

from group_consensus.models.types import (
    Opinion,
    Participant,
    PreferenceRanking,
    SessionConfig,
    Statement,
)


_RANKING_SYSTEM = """\
You are assessing how well a set of consensus statements represents a particular \
individual's perspective. Your job is to rank the statements from most to least \
acceptable to this individual, based on their expressed opinion.

Consider: Does the statement acknowledge what matters to them? Does it use language \
they would find fair? Would they feel their view is reflected in it, even if not \
perfectly captured?

Respond with only a numbered list of statement indices (1-based), most preferred first.
Example: 3, 1, 4, 2, 5
"""


def _build_ranking_prompt(
    participant: Participant,
    opinion: Opinion,
    candidates: list[Statement],
) -> str:
    lines = [
        f"Participant: {participant.name}",
        f"Their opinion: {opinion.text}",
        "",
        "Candidate consensus statements:",
    ]
    for i, stmt in enumerate(candidates, 1):
        lines.append(f"{i}. {stmt.text}")
    lines.append("")
    lines.append(
        "Rank these statements from most to least acceptable to this participant. "
        "Reply with only the numbers separated by commas."
    )
    return "\n".join(lines)


def _parse_ranking(raw: str, num_candidates: int) -> list[int]:
    """Parse '3, 1, 4, 2' → [2, 0, 3, 1] (0-based indices)."""
    # Extract all integers from the response
    tokens = [t.strip().rstrip(".),") for t in raw.replace("\n", ",").split(",")]
    seen: set[int] = set()
    ranked: list[int] = []
    for token in tokens:
        try:
            val = int(token)
            idx = val - 1  # convert to 0-based
            if 0 <= idx < num_candidates and idx not in seen:
                ranked.append(idx)
                seen.add(idx)
        except ValueError:
            continue

    # Append any missing candidates at the end (unranked = least preferred)
    for i in range(num_candidates):
        if i not in seen:
            ranked.append(i)

    return ranked


class RewardModel:
    """Predicts how each participant would rank a set of candidate statements."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.Anthropic()

    def predict_rankings(
        self,
        participants: list[Participant],
        opinions: list[Opinion],
        candidates: list[Statement],
        session_id: str,
        round_number: int,
    ) -> list[PreferenceRanking]:
        """
        For each participant, predict their ranking of the candidate statements.
        Returns one PreferenceRanking per participant.
        """
        opinion_map: dict[str, Opinion] = {op.participant_id: op for op in opinions}
        rankings: list[PreferenceRanking] = []

        # Build candidate context once, cache it
        candidate_lines = "\n".join(
            f"{i+1}. {s.text}" for i, s in enumerate(candidates)
        )
        cached_context = (
            f"Candidate consensus statements:\n{candidate_lines}\n\n"
            "You will be asked to rank these for different participants."
        )

        for participant in participants:
            opinion = opinion_map.get(participant.id)
            if opinion is None:
                # No opinion — assign neutral ordering
                rankings.append(
                    PreferenceRanking(
                        participant_id=participant.id,
                        statement_ids=[s.id for s in candidates],
                        session_id=session_id,
                        round_number=round_number,
                    )
                )
                continue

            prompt = _build_ranking_prompt(participant, opinion, candidates)

            response = self._client.messages.create(
                model=self.config.claude_model,
                max_tokens=256,
                system=[
                    {
                        "type": "text",
                        "text": _RANKING_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": cached_context,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            raw = response.content[0].text
            index_order = _parse_ranking(raw, len(candidates))
            statement_id_order: list[UUID] = [candidates[i].id for i in index_order]

            rankings.append(
                PreferenceRanking(
                    participant_id=participant.id,
                    statement_ids=statement_id_order,
                    session_id=session_id,
                    round_number=round_number,
                )
            )

        return rankings
