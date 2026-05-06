"""Tests for group-informed consensus detection."""

from uuid import uuid4

import pytest

from group_consensus.clustering.consensus import ConsensusDetector
from group_consensus.models.types import (
    Cluster,
    ClusteringResult,
    Statement,
    StatementType,
)


def make_stmt(text: str) -> Statement:
    return Statement(text=text, type=StatementType.CANDIDATE, session_id="s1")


def make_clustering(
    statements: list[Statement],
    approval_by_cluster: dict[str, dict[int, float]],
    num_clusters: int = 2,
) -> ClusteringResult:
    clusters = [
        Cluster(id=i, participant_ids=[f"p{i}"], centroid=[0.0])
        for i in range(num_clusters)
    ]
    return ClusteringResult(
        clusters=clusters,
        coordinates_2d={},
        num_clusters=num_clusters,
        silhouette_score=0.7,
        statement_approval_by_cluster=approval_by_cluster,
    )


def test_bridging_statement_detected():
    stmt = make_stmt("We all value transparency.")
    clustering = make_clustering(
        [stmt],
        {str(stmt.id): {0: 0.8, 1: 0.75}},
    )
    detector = ConsensusDetector(min_cluster_approval=0.6)
    result = detector.detect([stmt], clustering)

    assert len(result.bridging_statements) == 1
    assert result.bridging_statements[0].id == stmt.id
    assert len(result.divisive_statements) == 0


def test_divisive_statement_detected():
    stmt = make_stmt("We should fully embrace remote work.")
    clustering = make_clustering(
        [stmt],
        {str(stmt.id): {0: 0.9, 1: 0.1}},
    )
    detector = ConsensusDetector(min_cluster_approval=0.6)
    result = detector.detect([stmt], clustering)

    assert len(result.divisive_statements) == 1
    assert result.divisive_statements[0].id == stmt.id
    assert len(result.bridging_statements) == 0


def test_neither_bridging_nor_divisive():
    stmt = make_stmt("Some people like coffee.")
    clustering = make_clustering(
        [stmt],
        {str(stmt.id): {0: 0.5, 1: 0.45}},
    )
    detector = ConsensusDetector(min_cluster_approval=0.6)
    result = detector.detect([stmt], clustering)

    assert len(result.bridging_statements) == 0
    assert len(result.divisive_statements) == 0


def test_bridging_sorted_by_min_approval():
    s1 = make_stmt("Statement with lower min approval")
    s2 = make_stmt("Statement with higher min approval")
    clustering = make_clustering(
        [s1, s2],
        {
            str(s1.id): {0: 0.7, 1: 0.65},
            str(s2.id): {0: 0.9, 1: 0.85},
        },
    )
    detector = ConsensusDetector(min_cluster_approval=0.6)
    result = detector.detect([s1, s2], clustering)

    assert len(result.bridging_statements) == 2
    # s2 has higher min approval (0.85) so should come first
    assert result.bridging_statements[0].id == s2.id


def test_unknown_statement_id_skipped():
    """Statements not in the clustering result should be silently skipped."""
    stmt = make_stmt("Not in clustering")
    clustering = make_clustering([], {})
    detector = ConsensusDetector()
    result = detector.detect([stmt], clustering)
    assert result.bridging_statements == []
    assert result.divisive_statements == []
