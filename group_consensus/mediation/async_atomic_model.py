"""Async versions of AtomicStatementModel for notebook use."""

from __future__ import annotations

import json
import re

import anthropic

from group_consensus.mediation.atomic_model import (
    _COMPRESS_SYSTEM,
    _EXTRACT_SYSTEM,
    _SYNTHESISE_SYSTEM,
    _parse_numbered_list,
)
from group_consensus.models.types import (
    CompressionResult,
    MergeRecord,
    Opinion,
    Participant,
    SessionConfig,
    Statement,
    StatementType,
)


class AsyncAtomicStatementModel:
    """Async atomic extraction, compression, and synthesis."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.AsyncAnthropic()

    async def extract(
        self,
        topic: str,
        opinions: list[Opinion],
        participants: list[Participant],
    ) -> list[Statement]:
        name_map = {p.id: p.name for p in participants}
        opinion_lines = "\n".join(
            f"- {name_map.get(o.participant_id, o.participant_id)}: {o.text}"
            for o in opinions
        )

        response = await self._client.messages.create(
            model=self.config.claude_model,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": _EXTRACT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Topic: {topic}\n\nOpinions:\n{opinion_lines}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all atomic statements implied by these opinions. "
                            "Number each one. 5–15 words each, declarative form."
                        ),
                    },
                ],
            }],
        )

        texts = _parse_numbered_list(response.content[0].text)
        return [
            Statement(
                text=t,
                type=StatementType.ATOMIC,
                session_id=self.config.session_id,
            )
            for t in texts
        ]

    async def compress(
        self,
        topic: str,
        statements: list[Statement],
    ) -> CompressionResult:
        numbered = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(statements))

        response = await self._client.messages.create(
            model=self.config.claude_model,
            max_tokens=4096,
            system=_COMPRESS_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Atomic statements to compress:\n{numbered}\n\n"
                    "Return the result as JSON."
                ),
            }],
        )

        raw = response.content[0].text
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else json.loads(raw)

        compressed: list[Statement] = []
        for item in data.get("statements", []):
            stype = StatementType.CONTESTED if item.get("contested") else StatementType.ATOMIC
            compressed.append(Statement(
                text=item["text"],
                type=stype,
                session_id=self.config.session_id,
            ))

        merges = [
            MergeRecord(
                kept=m["kept"],
                absorbed=m.get("absorbed", []),
                reason=m.get("reason", ""),
            )
            for m in data.get("merges", [])
        ]

        contested_pairs = [
            (pair[0], pair[1])
            for pair in data.get("contested_pairs", [])
            if len(pair) == 2
        ]

        return CompressionResult(
            statements=compressed,
            merges=merges,
            contested_pairs=contested_pairs,
        )

    async def synthesise(
        self,
        topic: str,
        winning_statements: list[Statement],
    ) -> str:
        atoms = "\n".join(f"- {s.text}" for s in winning_statements)

        response = await self._client.messages.create(
            model=self.config.claude_model,
            max_tokens=1024,
            system=_SYNTHESISE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Agreed atomic statements:\n{atoms}\n\n"
                    "Write the group consensus statement."
                ),
            }],
        )

        return response.content[0].text.strip()
