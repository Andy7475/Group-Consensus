"""Async versions of mediation components for notebook and CLI use.

Key difference from the sync versions: reward model predictions for all
participants run concurrently via asyncio.gather, giving roughly N× speedup
where N = number of participants. A semaphore caps concurrency to avoid
hitting API rate limits.

Usage in a Jupyter notebook cell:
    result = await mediator.run(topic, participants, opinions)

Or from a regular script (with nest_asyncio applied):
    result = asyncio.run(mediator.run(topic, participants, opinions))
"""

from __future__ import annotations

import asyncio

import anthropic

from group_consensus.mediation.mediator import MediationResult
from group_consensus.mediation.reward_model import (
    _RANKING_SYSTEM,
    _build_ranking_prompt,
    _parse_ranking,
)
from group_consensus.mediation.social_choice import select_winner_from_rankings
from group_consensus.mediation.statement_model import (
    _SYSTEM_PROMPT,
    _format_opinions,
    _parse_statements,
)
from group_consensus.models.types import (
    DeliberationRound,
    Opinion,
    Participant,
    PreferenceRanking,
    SessionConfig,
    Statement,
    StatementType,
)

_MAX_CONCURRENT_REQUESTS = 5  # stays well within Anthropic rate limits


class AsyncStatementModel:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic()

    async def generate_candidates(
        self,
        topic: str,
        opinions: list[Opinion],
        participants: list[Participant],
        critiques: list[Opinion] | None = None,
        previous_winner: Statement | None = None,
        round_number: int = 0,
    ) -> list[Statement]:
        n = self.config.num_candidate_statements
        opinion_text = _format_opinions(opinions, participants)

        user_content: list[dict] = [
            {
                "type": "text",
                "text": f"Topic: {topic}\n\nParticipant opinions:\n{opinion_text}\n\n",
                "cache_control": {"type": "ephemeral"},
            }
        ]

        if previous_winner and critiques:
            critique_text = _format_opinions(critiques, participants)
            user_content.append({
                "type": "text",
                "text": (
                    f"Previous consensus statement:\n\"{previous_winner.text}\"\n\n"
                    f"Participant critiques:\n{critique_text}\n\n"
                ),
            })

        user_content.append({
            "type": "text",
            "text": (
                f"Generate exactly {n} distinct candidate consensus statements. "
                "Number each one. Each should be one or two sentences, "
                "taking a different angle or emphasis on the common ground."
            ),
        })

        response = await self._client.messages.create(
            model=self.config.claude_model,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
        )

        texts = _parse_statements(response.content[0].text)[:n]
        while len(texts) < n:
            texts.append(f"The group acknowledges complex views on this topic. (fallback {len(texts) + 1})")

        return [
            Statement(
                text=text,
                type=StatementType.CANDIDATE,
                session_id=self.config.session_id,
                round_number=round_number,
            )
            for text in texts
        ]


class AsyncRewardModel:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic()
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    async def _rank_one_participant(
        self,
        participant: Participant,
        opinion: Opinion | None,
        candidates: list[Statement],
        candidate_context: str,
        session_id: str,
        round_number: int,
    ) -> PreferenceRanking:
        if opinion is None:
            return PreferenceRanking(
                participant_id=participant.id,
                statement_ids=[s.id for s in candidates],
                session_id=session_id,
                round_number=round_number,
            )

        prompt = _build_ranking_prompt(participant, opinion, candidates)

        async with self._sem:
            response = await self._client.messages.create(
                model=self.config.claude_model,
                max_tokens=256,
                system=[{
                    "type": "text",
                    "text": _RANKING_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": candidate_context,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

        index_order = _parse_ranking(response.content[0].text, len(candidates))
        return PreferenceRanking(
            participant_id=participant.id,
            statement_ids=[candidates[i].id for i in index_order],
            session_id=session_id,
            round_number=round_number,
        )

    async def predict_rankings(
        self,
        participants: list[Participant],
        opinions: list[Opinion],
        candidates: list[Statement],
        session_id: str,
        round_number: int,
    ) -> list[PreferenceRanking]:
        opinion_map = {op.participant_id: op for op in opinions}

        # Build candidate context once — cached across all participant calls
        candidate_context = (
            "Candidate consensus statements:\n"
            + "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(candidates))
            + "\n\nYou will be asked to rank these for different participants."
        )

        tasks = [
            self._rank_one_participant(
                p, opinion_map.get(p.id), candidates,
                candidate_context, session_id, round_number,
            )
            for p in participants
        ]
        return list(await asyncio.gather(*tasks))


class AsyncMediator:
    """
    Async deliberation loop for use in Jupyter notebooks and async scripts.

    All participant ranking predictions run concurrently, giving roughly
    N× speedup over the synchronous Mediator (N = participant count).
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._statement_model = AsyncStatementModel(config)
        self._reward_model = AsyncRewardModel(config)

    async def run(
        self,
        topic: str,
        participants: list[Participant],
        opinions: list[Opinion],
        critique_fn=None,
        verbose: bool = True,
    ) -> MediationResult:
        rounds: list[DeliberationRound] = []
        previous_winner: Statement | None = None
        critiques: list[Opinion] = []
        current_opinions = list(opinions)

        for round_num in range(self.config.max_deliberation_rounds):
            if verbose:
                print(f"\n── Round {round_num + 1} {'─' * 40}")
                print(f"   Generating {self.config.num_candidate_statements} candidate statements...")

            candidates = await self._statement_model.generate_candidates(
                topic=topic,
                opinions=current_opinions,
                participants=participants,
                critiques=critiques if critiques else None,
                previous_winner=previous_winner,
                round_number=round_num,
            )

            if verbose:
                print(f"   Ranking candidates for {len(participants)} participants (concurrent)...")

            rankings = await self._reward_model.predict_rankings(
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

            if verbose:
                preview = winner.text[:90] + "..." if len(winner.text) > 90 else winner.text
                print(f"   → Schulze winner: \"{preview}\"")

            new_critiques: list[Opinion] = []
            if critique_fn is not None:
                new_critiques = critique_fn(winner, participants)

            rounds.append(DeliberationRound(
                round_number=round_num,
                candidate_statements=candidates,
                rankings=rankings,
                winning_statement=winner,
                critiques=new_critiques,
            ))

            previous_winner = winner
            critiques = new_critiques

            if not new_critiques:
                if verbose:
                    print("   No critiques submitted — loop converged.")
                break

        final = rounds[-1].winning_statement
        final.type = StatementType.REFINED

        if verbose:
            print(f"\n{'═' * 55}")
            print(f"Mediation complete ({len(rounds)} round{'s' if len(rounds) != 1 else ''})")
            print(f"{'═' * 55}")
            print(f"\"{final.text}\"")
            print(f"{'═' * 55}\n")

        return MediationResult(
            consensus_statement=final,
            rounds=rounds,
            session_id=self.config.session_id,
            topic=topic,
        )
