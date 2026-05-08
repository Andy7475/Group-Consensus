"""Atomic statement extraction, compression, and synthesis.

Replaces the comprehensive-statement approach from the original Habermas
Machine with shorter, single-idea statements suited to Pol.is-style voting.

Three steps:
  1. extract()   — pull one-idea statements from all opinions (~20-30 raw)
  2. compress()  — merge synonyms and compatible constraints; flag contradictions
  3. synthesise()— take the winning atomic statements and write a group statement

The compression step is where the important logic lives. The LLM merges
semantic equivalents ("loud" / "noisy") and compatible constraints ("by 8pm"
subsumes "by 9pm") but must NOT resolve genuine contradictions — those stay
as separate CONTESTED statements for Pol.is voting to resolve.
"""

from __future__ import annotations

import json
import re

import anthropic

from group_consensus.models.types import (
    CompressionResult,
    MergeRecord,
    Opinion,
    Participant,
    SessionConfig,
    Statement,
    StatementType,
)


_EXTRACT_SYSTEM = """\
You extract atomic statements from group opinions for deliberation.

An atomic statement:
- Expresses exactly one idea
- Is 5–15 words
- Is declarative third-person: "The venue has step-free access"
- Can be clearly agreed or disagreed with on its own
- Contains no compound ideas ("central AND has parking" → two statements)

Extract everything — practical constraints, preferences, experience qualities,
things people said would make the option unworkable. Include implicit
requirements, not just explicit ones. Better to over-extract at this stage.
"""

_COMPRESS_SYSTEM = """\
You compress a list of atomic statements by merging equivalents.

Rules:

MERGE when statements mean the same thing or one subsumes the other:
  - Semantic synonyms: "loud" and "noisy" → one statement about noise
  - Compatible constraints: "finish by 8pm" and "finish by 9pm" →
    keep "finish by 8pm" (the 8pm person needs this; the 9pm person is
    also satisfied by it)
  - Compatible requirements: "vegetarian options" and "vegan options" →
    "plant-based food options are available"

DO NOT MERGE genuine contradictions. Mark them CONTESTED and keep both:
  - "there should be a team activity" vs "no structured activities"
  - "drinks covered by company" vs "the event is non-alcoholic"
  - These are resolved by voting, not by you

Return valid JSON with this structure:
{
  "statements": [
    {"text": "...", "contested": false},
    {"text": "...", "contested": true},
    ...
  ],
  "merges": [
    {"kept": "...", "absorbed": ["...", "..."], "reason": "..."},
    ...
  ],
  "contested_pairs": [
    ["statement text A", "statement text B"],
    ...
  ]
}
"""

_SYNTHESISE_SYSTEM = """\
You write a group consensus document from a set of agreed atomic statements.

The atomic statements below have achieved broad agreement across all opinion
groups in the deliberation. Your job is to write a coherent group statement
that honours every point on the list.

Write naturally — not a bullet list flattened into prose. Be specific enough
to be actionable. For complex topics you may write several paragraphs or use
light structure (e.g. short headed sections). Do not add anything that is not
implied by the atomic statements.
"""


def _parse_numbered_list(raw: str) -> list[str]:
    """Extract text from a numbered or bulleted list in LLM output."""
    lines = raw.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        match = re.match(r"^(?:\d+[.)]\s*|[-*•]\s*)(.+)$", line)
        if match:
            text = match.group(1).strip().rstrip(".")
            if 4 < len(text) < 200:
                results.append(text)
    return results


class AtomicStatementModel:
    """Extracts, compresses, and synthesises atomic statements."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._client = anthropic.Anthropic()

    def extract(
        self,
        topic: str,
        opinions: list[Opinion],
        participants: list[Participant],
    ) -> list[Statement]:
        """
        Pull raw atomic statements from all opinions.
        Returns 15–30 statements before deduplication.
        """
        name_map = {p.id: p.name for p in participants}
        opinion_lines = "\n".join(
            f"- {name_map.get(o.participant_id, o.participant_id)}: {o.text}"
            for o in opinions
        )

        response = self._client.messages.create(
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

    def compress(
        self,
        topic: str,
        statements: list[Statement],
    ) -> CompressionResult:
        """
        Merge semantic equivalents and compatible constraints.
        Contradictions are kept as paired CONTESTED statements.
        """
        numbered = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(statements))

        response = self._client.messages.create(
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
        # Extract JSON block from response (LLM sometimes wraps it in markdown)
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

    def synthesise(
        self,
        topic: str,
        winning_statements: list[Statement],
    ) -> str:
        """
        Generate a comprehensive group statement from winning atomic statements.
        Returns plain text — could be a sentence, a paragraph, or a structured
        brief depending on the number and complexity of the inputs.
        """
        atoms = "\n".join(f"- {s.text}" for s in winning_statements)

        response = self._client.messages.create(
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
