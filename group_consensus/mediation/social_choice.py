"""Schulze method social-choice voting for consensus statement selection.

The Schulze method is a Condorcet voting method that finds the candidate with the
strongest pairwise paths to all other candidates. It is used here to aggregate
predicted preference rankings from the reward model into a single winning statement.

Reference: Schulze, M. (2011). A new monotonic, clone-independent, reversal symmetric,
and condorcet-consistent single-winner election method. Social Choice and Welfare, 36(2).
"""

from __future__ import annotations

from uuid import UUID


def build_pairwise_matrix(
    rankings: list[list[int]], num_candidates: int
) -> list[list[int]]:
    """Build d[i][j] = number of voters preferring candidate i over candidate j."""
    d: list[list[int]] = [[0] * num_candidates for _ in range(num_candidates)]
    for ranking in rankings:
        # ranking is an ordered list of candidate indices, best first
        for pos_i, candidate_i in enumerate(ranking):
            for candidate_j in ranking[pos_i + 1 :]:
                d[candidate_i][candidate_j] += 1
    return d


def strongest_paths(
    d: list[list[int]], num_candidates: int
) -> list[list[int]]:
    """
    Floyd-Warshall to find the strongest path between every pair of candidates.
    Path strength is the minimum edge weight along the path.
    Edge i→j exists (with weight d[i][j]) only when d[i][j] > d[j][i].
    """
    p: list[list[int]] = [[0] * num_candidates for _ in range(num_candidates)]

    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                p[i][j] = d[i][j] if d[i][j] > d[j][i] else 0

    for k in range(num_candidates):
        for i in range(num_candidates):
            if i == k:
                continue
            for j in range(num_candidates):
                if j == i or j == k:
                    continue
                p[i][j] = max(p[i][j], min(p[i][k], p[k][j]))

    return p


def schulze_ranking(rankings: list[list[int]], num_candidates: int) -> list[int]:
    """
    Return candidate indices ordered from winner to last place using the Schulze method.

    Args:
        rankings: Each element is an ordered list of candidate indices (0-based),
                  most preferred first. May be partial rankings.
        num_candidates: Total number of candidates.

    Returns:
        Candidate indices sorted from Schulze winner to last place.
    """
    if num_candidates == 0:
        return []
    if num_candidates == 1:
        return [0]

    d = build_pairwise_matrix(rankings, num_candidates)
    p = strongest_paths(d, num_candidates)

    # wins[i] = number of other candidates that i beats via strongest path
    wins = [
        sum(1 for j in range(num_candidates) if i != j and p[i][j] > p[j][i])
        for i in range(num_candidates)
    ]

    return sorted(range(num_candidates), key=lambda i: wins[i], reverse=True)


def select_winner_from_rankings(
    statement_ids: list[UUID],
    participant_rankings: list[list[UUID]],
) -> UUID:
    """
    Given a list of statements and participants' ranked preferences over them,
    return the UUID of the Schulze winner.

    Args:
        statement_ids: All candidate statement UUIDs.
        participant_rankings: Each element is a participant's preference order
                              (list of statement UUIDs, most preferred first).
    """
    id_to_idx = {sid: i for i, sid in enumerate(statement_ids)}

    # Convert UUID rankings to integer index rankings, skipping unknowns
    index_rankings: list[list[int]] = []
    for ranking in participant_rankings:
        index_rankings.append([id_to_idx[sid] for sid in ranking if sid in id_to_idx])

    ordered = schulze_ranking(index_rankings, len(statement_ids))
    return statement_ids[ordered[0]]
