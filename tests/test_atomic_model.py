"""Tests for atomic statement parsing and compression logic.

These tests cover the parsing utilities only — the LLM calls themselves
require an API key and are not unit-tested here.
"""

import json

import pytest

from group_consensus.mediation.atomic_model import _parse_numbered_list
from group_consensus.models.types import (
    CompressionResult,
    MergeRecord,
    Statement,
    StatementType,
)


def test_parse_numbered_list_standard():
    raw = "1. The venue has step-free access\n2. The event finishes by 8pm\n3. Food costs are covered"
    result = _parse_numbered_list(raw)
    assert len(result) == 3
    assert result[0] == "The venue has step-free access"
    assert result[1] == "The event finishes by 8pm"


def test_parse_numbered_list_with_parens():
    raw = "1) First statement\n2) Second statement"
    result = _parse_numbered_list(raw)
    assert len(result) == 2


def test_parse_numbered_list_strips_trailing_period():
    raw = "1. The venue is central.\n2. Drinks are covered."
    result = _parse_numbered_list(raw)
    assert not result[0].endswith(".")
    assert not result[1].endswith(".")


def test_parse_numbered_list_skips_short_lines():
    raw = "1. Ok\n2. The venue has step-free access throughout"
    result = _parse_numbered_list(raw)
    assert len(result) == 1
    assert "step-free" in result[0]


def test_parse_numbered_list_handles_bullets():
    raw = "- The venue is accessible\n- The event is fully funded\n* Noise level is low"
    result = _parse_numbered_list(raw)
    assert len(result) == 3


def test_parse_numbered_list_empty():
    assert _parse_numbered_list("") == []
    assert _parse_numbered_list("No list here, just prose.") == []


def test_statement_type_atomic_and_contested():
    # Verify the new types exist and are distinct
    assert StatementType.ATOMIC != StatementType.CONTESTED
    assert StatementType.ATOMIC == "atomic"
    assert StatementType.CONTESTED == "contested"


def test_compression_result_model():
    stmts = [
        Statement(text="The event finishes by 8pm", type=StatementType.ATOMIC, session_id="s1"),
        Statement(text="There is a team activity", type=StatementType.CONTESTED, session_id="s1"),
        Statement(text="No structured activities", type=StatementType.CONTESTED, session_id="s1"),
    ]
    merges = [
        MergeRecord(
            kept="The event finishes by 8pm",
            absorbed=["The event finishes by 9pm"],
            reason="8pm is stricter and satisfies both requirements",
        )
    ]
    result = CompressionResult(
        statements=stmts,
        merges=merges,
        contested_pairs=[("There is a team activity", "No structured activities")],
    )

    assert len(result.statements) == 3
    assert len(result.merges) == 1
    assert result.merges[0].absorbed == ["The event finishes by 9pm"]
    assert len(result.contested_pairs) == 1


def test_merge_record_reason_preserved():
    m = MergeRecord(
        kept="The venue is accessible",
        absorbed=["Step-free access required", "Wheelchair accessible"],
        reason="All three express the same accessibility requirement",
    )
    assert "accessibility" in m.reason
    assert len(m.absorbed) == 2
