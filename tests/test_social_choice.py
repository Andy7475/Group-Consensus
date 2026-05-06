"""Tests for the Schulze social-choice voting implementation."""

from uuid import uuid4

import pytest

from group_consensus.mediation.social_choice import (
    build_pairwise_matrix,
    schulze_ranking,
    select_winner_from_rankings,
    strongest_paths,
)


def test_single_candidate():
    result = schulze_ranking([], 1)
    assert result == [0]


def test_two_candidates_unanimous():
    # Both voters prefer 0 over 1
    rankings = [[0, 1], [0, 1]]
    result = schulze_ranking(rankings, 2)
    assert result[0] == 0


def test_two_candidates_split():
    # 2 prefer 0, 1 prefers 1 → 0 wins
    rankings = [[0, 1], [0, 1], [1, 0]]
    result = schulze_ranking(rankings, 3)
    assert result[0] == 0


def test_condorcet_cycle_resolved():
    # Classic Condorcet cycle: A>B, B>C, C>A (each by 2 voters)
    # Schulze resolves by strongest path
    # Voter 1: A>B>C, Voter 2: B>C>A, Voter 3: C>A>B
    rankings = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    result = schulze_ranking(rankings, 3)
    # All three are tied in a cycle, Schulze should still return 3 candidates
    assert len(result) == 3
    assert set(result) == {0, 1, 2}


def test_clear_winner():
    # Candidate 2 is everyone's first choice
    rankings = [[2, 0, 1], [2, 1, 0], [2, 0, 1]]
    result = schulze_ranking(rankings, 3)
    assert result[0] == 2


def test_pairwise_matrix():
    # rankings: voter 1 prefers 0>1>2, voter 2 prefers 1>0>2
    rankings = [[0, 1, 2], [1, 0, 2]]
    d = build_pairwise_matrix(rankings, 3)
    assert d[0][1] == 1  # one voter prefers 0 over 1
    assert d[1][0] == 1  # one voter prefers 1 over 0
    assert d[0][2] == 2  # both voters prefer 0 over 2
    assert d[1][2] == 2  # both voters prefer 1 over 2


def test_select_winner_from_uuid_rankings():
    ids = [uuid4() for _ in range(3)]
    # All participants prefer ids[1]
    participant_rankings = [
        [ids[1], ids[0], ids[2]],
        [ids[1], ids[2], ids[0]],
    ]
    winner = select_winner_from_rankings(ids, participant_rankings)
    assert winner == ids[1]


def test_partial_rankings_handled():
    # Partial rankings should not crash
    rankings = [[0, 2], [1], [2, 0, 1]]
    result = schulze_ranking(rankings, 3)
    assert len(result) == 3


def test_empty_rankings():
    # No preference data → still returns all candidates
    result = schulze_ranking([], 3)
    assert len(result) == 3
    assert set(result) == {0, 1, 2}
