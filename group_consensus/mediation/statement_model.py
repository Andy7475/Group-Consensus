"""LLM-based consensus statement generation using the Claude API.

Mirrors the statement generation phase of the Habermas Machine:
given a set of participant opinions on a topic, generate N candidate
consensus statements that attempt to reflect the group's shared values
while acknowledging genuine disagreements.

Uses Anthropic prompt caching to avoid re-sending the topic context and
participant opinions on every generation call within a session.
"""

from __future__ import annotations

import re

import anthropic

from group_consensus.models.types import Opinion, Participant, SessionConfig, Statement, StatementType


_SYSTEM_PROMPT = """\
You are an expert facilitator helping a group find common ground on a difficult topic.
Your task is to craft consensus statements — not summaries of majority opinion, but \
statements that genuinely reflect what people across the group can agree on, including \
their shared concerns, values, and areas of honest uncertainty.

A good consensus statement:
- Acknowledges tensions rather than papering over them
- Uses language all participants could recognise as fair
- Does not erase minority views but finds formulations they can accept
- Is specific enough to be actionable but broad enough to unite
- Is written in plain language, not bureaucratic or hedged beyond meaning
"""


def _format_opinions(opinions: list[Opinion], participants: list[Participant]) -> str:
    name_map = {p.id: p.name for p in participants}
    lines = []
    for op in opinions:
        name = name_map.get(op.participant_id, op.participant_id)
        lines.append(f"- {name}: {op.text}")
    return "\n".join(lines)


def _parse_statements(raw: str) -> list[str]:
    """Extract numbered or bulleted statements from LLM output."""
    lines = raw.strip().split("\n")
    statements = []
    for line in lines:
        line = line.strip()
        # Match "1. ...", "1) ...", "- ...", "* ..."
        match = re.match(r"^(?:\d+[.)]\s*|[-*]\s*)(.+)$", line)
        if match:
            text = match.group(1).strip()
            if len(text) > 20:
                statements.append(text)
    return statements


class StatementModel:
    """Generates candidate consensus statements from participant opinions."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.Anthropic()

    def generate_candidates(
        self,
        topic: str,
        opinions: list[Opinion],
        participants: list[Participant],
        critiques: list[Opinion] | None = None,
        previous_winner: Statement | None = None,
        round_number: int = 0,
    ) -> list[Statement]:
        """
        Generate `config.num_candidate_statements` candidate consensus statements.

        Uses prompt caching on the system prompt and opinion list so that
        multiple generation calls within a session are cheaper.
        """
        n = self.config.num_candidate_statements
        opinion_text = _format_opinions(opinions, participants)

        user_content: list[anthropic.types.ContentBlockParam] = []

        # The topic + all opinions are the stable context — cache them
        user_content.append({
            "type": "text",
            "text": (
                f"Topic: {topic}\n\n"
                f"Participant opinions:\n{opinion_text}\n\n"
            ),
            "cache_control": {"type": "ephemeral"},
        })

        if previous_winner and critiques:
            critique_text = _format_opinions(critiques, participants)
            user_content.append({
                "type": "text",
                "text": (
                    f"The previous consensus statement was:\n\"{previous_winner.text}\"\n\n"
                    f"Participants offered these critiques:\n{critique_text}\n\n"
                ),
            })

        user_content.append({
            "type": "text",
            "text": (
                f"Please generate exactly {n} distinct candidate consensus statements.\n"
                "Number each one. Each should be a single sentence or two short sentences. "
                "They should each take a different angle or emphasis on the common ground."
            ),
        })

        response = self._client.messages.create(
            model=self.config.claude_model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text
        texts = _parse_statements(raw)

        # Trim or pad to exactly n candidates
        texts = texts[:n]
        while len(texts) < n:
            texts.append(f"The group acknowledges complex views on {topic}. (fallback {len(texts)+1})")

        return [
            Statement(
                text=text,
                type=StatementType.CANDIDATE,
                session_id=self.config.session_id,
                round_number=round_number,
            )
            for text in texts
        ]
