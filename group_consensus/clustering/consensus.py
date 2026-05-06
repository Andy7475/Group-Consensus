"""Group-informed consensus detection.

A statement achieves group-informed consensus when every opinion cluster
approves of it above the threshold — not just a majority overall.
This is the key insight from Polis: consensus across divides is more
valuable than majority approval.

Divisive statements (high variance across clusters) are also surfaced
because they reveal where facilitation energy is most needed.
"""

from __future__ import annotations

from group_consensus.clustering.engine import ClusteringEngine
from group_consensus.clustering.opinion_matrix import OpinionMatrix
from group_consensus.models.types import (
    ClusteringResult,
    ConsensusResult,
    Statement,
)


class ConsensusDetector:
    """Identifies bridging and divisive statements from a clustering result."""

    def __init__(self, min_cluster_approval: float = 0.6) -> None:
        self.min_cluster_approval = min_cluster_approval

    def detect(
        self,
        statements: list[Statement],
        clustering: ClusteringResult,
    ) -> ConsensusResult:
        """
        Classify statements as bridging (group-informed consensus) or divisive.

        A statement is bridging if every cluster approves it at >= min_cluster_approval.
        A statement is divisive if at least one cluster strongly rejects it (< 0.3)
        while another strongly approves it (> 0.7).
        """
        stmt_map = {str(s.id): s for s in statements}
        bridging: list[Statement] = []
        divisive: list[Statement] = []

        for sid_str, cluster_approvals in clustering.statement_approval_by_cluster.items():
            if sid_str not in stmt_map:
                continue
            stmt = stmt_map[sid_str]
            approvals = list(cluster_approvals.values())

            if not approvals:
                continue

            all_above_threshold = all(a >= self.min_cluster_approval for a in approvals)
            has_high_variance = (max(approvals) - min(approvals)) > 0.4

            if all_above_threshold:
                bridging.append(stmt)
            elif has_high_variance and any(a < 0.3 for a in approvals):
                divisive.append(stmt)

        # Sort bridging by minimum cluster approval (most universally accepted first)
        def min_approval(s: Statement) -> float:
            approvals = clustering.statement_approval_by_cluster.get(str(s.id), {})
            return min(approvals.values()) if approvals else 0.0

        bridging.sort(key=min_approval, reverse=True)

        # Sort divisive by variance (most divisive first)
        def approval_variance(s: Statement) -> float:
            approvals = list(
                clustering.statement_approval_by_cluster.get(str(s.id), {}).values()
            )
            if not approvals:
                return 0.0
            mean = sum(approvals) / len(approvals)
            return sum((a - mean) ** 2 for a in approvals) / len(approvals)

        divisive.sort(key=approval_variance, reverse=True)

        return ConsensusResult(
            bridging_statements=bridging,
            divisive_statements=divisive,
            clustering=clustering,
            group_consensus_threshold=self.min_cluster_approval,
        )

    @classmethod
    def from_votes(
        cls,
        statements: list[Statement],
        opinion_matrix: OpinionMatrix,
        max_clusters: int = 5,
        min_cluster_approval: float = 0.6,
    ) -> ConsensusResult:
        """Convenience method: cluster then detect consensus in one call."""
        engine = ClusteringEngine(max_clusters=max_clusters)
        clustering = engine.run(opinion_matrix)
        detector = cls(min_cluster_approval=min_cluster_approval)
        return detector.detect(statements, clustering)
