"""Tests for opinion matrix and clustering engine."""

from uuid import uuid4

import numpy as np
import pytest

from group_consensus.clustering.engine import ClusteringEngine
from group_consensus.clustering.opinion_matrix import OpinionMatrix
from group_consensus.models.types import Statement, StatementType, Vote, VoteValue


def make_statement(session_id: str = "s1") -> Statement:
    return Statement(
        text="Test statement",
        type=StatementType.CANDIDATE,
        session_id=session_id,
    )


def test_opinion_matrix_basic():
    matrix = OpinionMatrix()
    stmt = make_statement()
    vote = Vote(
        participant_id="p1",
        statement_id=stmt.id,
        value=VoteValue.AGREE,
        session_id="s1",
    )
    matrix.add_statement(stmt)
    matrix.add_vote(vote)

    arr = matrix.to_numpy()
    assert arr.shape == (1, 1)
    assert arr[0, 0] == 1.0


def test_opinion_matrix_multiple_votes():
    matrix = OpinionMatrix()
    stmts = [make_statement() for _ in range(3)]
    for s in stmts:
        matrix.add_statement(s)

    votes = [
        Vote(participant_id="p1", statement_id=stmts[0].id, value=VoteValue.AGREE, session_id="s1"),
        Vote(participant_id="p1", statement_id=stmts[1].id, value=VoteValue.DISAGREE, session_id="s1"),
        Vote(participant_id="p2", statement_id=stmts[0].id, value=VoteValue.AGREE, session_id="s1"),
        Vote(participant_id="p2", statement_id=stmts[2].id, value=VoteValue.PASS, session_id="s1"),
    ]
    for v in votes:
        matrix.add_vote(v)

    arr = matrix.to_numpy()
    assert arr.shape == (2, 3)


def test_approval_rates():
    matrix = OpinionMatrix()
    stmt = make_statement()
    matrix.add_statement(stmt)

    for pid in ["p1", "p2", "p3"]:
        val = VoteValue.AGREE if pid != "p3" else VoteValue.DISAGREE
        matrix.add_vote(
            Vote(participant_id=pid, statement_id=stmt.id, value=val, session_id="s1")
        )

    rates = matrix.statement_approval_rates()
    # 2 agree, 1 disagree → 2/3 ≈ 0.667
    assert abs(rates[stmt.id] - 2 / 3) < 0.01


def test_next_statements_unseen():
    matrix = OpinionMatrix()
    stmts = [make_statement() for _ in range(5)]
    for s in stmts:
        matrix.add_statement(s)
        matrix.add_participant("p1")

    # p1 votes on only first 2
    for s in stmts[:2]:
        matrix.add_vote(
            Vote(participant_id="p1", statement_id=s.id, value=VoteValue.AGREE, session_id="s1")
        )

    unseen = matrix.next_statements_for_participant("p1", n=10)
    # Should return the 3 unseen statements
    assert len(unseen) == 3
    assert all(sid not in [s.id for s in stmts[:2]] for sid in unseen)


def test_clustering_engine_two_groups():
    """Create two clearly separated opinion groups and verify clustering finds 2."""
    matrix = OpinionMatrix()
    stmts = [make_statement() for _ in range(6)]
    for s in stmts:
        matrix.add_statement(s)

    # Group A: strongly agrees with first 3 statements, disagrees with last 3
    group_a = [f"a{i}" for i in range(8)]
    # Group B: opposite pattern
    group_b = [f"b{i}" for i in range(8)]

    for pid in group_a:
        for i, s in enumerate(stmts):
            val = VoteValue.AGREE if i < 3 else VoteValue.DISAGREE
            matrix.add_vote(Vote(participant_id=pid, statement_id=s.id, value=val, session_id="s1"))

    for pid in group_b:
        for i, s in enumerate(stmts):
            val = VoteValue.DISAGREE if i < 3 else VoteValue.AGREE
            matrix.add_vote(Vote(participant_id=pid, statement_id=s.id, value=val, session_id="s1"))

    engine = ClusteringEngine(max_clusters=4)
    result = engine.run(matrix)

    assert result.num_clusters == 2
    assert result.silhouette_score > 0.5

    # Every participant should have 2D coordinates
    all_pids = group_a + group_b
    for pid in all_pids:
        assert pid in result.coordinates_2d
        assert len(result.coordinates_2d[pid]) == 2


def test_clustering_engine_too_few_participants():
    """With fewer than 4 participants, should return 1 cluster without crashing."""
    matrix = OpinionMatrix()
    stmt = make_statement()
    matrix.add_statement(stmt)

    for pid in ["p1", "p2"]:
        matrix.add_vote(
            Vote(participant_id=pid, statement_id=stmt.id, value=VoteValue.AGREE, session_id="s1")
        )

    engine = ClusteringEngine(max_clusters=4)
    result = engine.run(matrix)
    assert result.num_clusters == 1
