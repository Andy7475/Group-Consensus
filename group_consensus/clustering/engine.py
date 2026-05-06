"""K-means clustering and dimensionality reduction for the opinion matrix.

Follows the Polis approach:
- Try k=2..max_k clusters using K-means
- Select k by silhouette score
- Reduce to 2D with PCA for visualisation

The output is a ClusteringResult containing cluster assignments, 2D coordinates,
and per-cluster statement approval rates.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from group_consensus.clustering.opinion_matrix import OpinionMatrix
from group_consensus.models.types import Cluster, ClusteringResult


def _best_k(matrix: np.ndarray, max_k: int) -> tuple[int, float, np.ndarray]:
    """Find the k with the best silhouette score. Returns (k, score, labels)."""
    if matrix.shape[0] < 4:
        # Too few participants to cluster meaningfully
        labels = np.zeros(matrix.shape[0], dtype=int)
        return 1, 0.0, labels

    best_k = 2
    best_score = -1.0
    best_labels = np.zeros(matrix.shape[0], dtype=int)

    for k in range(2, min(max_k + 1, matrix.shape[0])):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)
        # silhouette_score requires at least 2 distinct labels
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_k, best_score, best_labels


class ClusteringEngine:
    """Clusters participants and produces visualisation coordinates."""

    def __init__(self, max_clusters: int = 5) -> None:
        self.max_clusters = max_clusters

    def run(self, opinion_matrix: OpinionMatrix) -> ClusteringResult:
        """
        Cluster participants and compute 2D projections.

        Returns a ClusteringResult with cluster assignments, 2D coordinates,
        and per-cluster approval rates for every statement.
        """
        participant_ids = opinion_matrix.participant_ids
        statement_ids = opinion_matrix.statement_ids

        if not participant_ids or not statement_ids:
            raise ValueError("Opinion matrix is empty — add participants and votes first.")

        matrix = opinion_matrix.to_numpy()

        # Standardise before clustering (not before PCA — keep PCA on raw votes)
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)

        num_k, sil_score, labels = _best_k(matrix_scaled, self.max_clusters)

        # 2D PCA on the raw (unscaled) matrix for interpretable visualisation
        n_components = min(2, matrix.shape[1], matrix.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        coords_2d = pca.fit_transform(matrix)

        if n_components == 1:
            # Pad with zeros if only one component possible
            coords_2d = np.hstack([coords_2d, np.zeros((coords_2d.shape[0], 1))])

        # Build Cluster objects
        clusters: list[Cluster] = []
        for k in range(num_k):
            member_indices = np.where(labels == k)[0]
            member_ids = [participant_ids[i] for i in member_indices]
            centroid = matrix_scaled[member_indices].mean(axis=0).tolist()
            clusters.append(
                Cluster(id=k, participant_ids=member_ids, centroid=centroid)
            )

        # 2D coordinates per participant
        coordinates_2d: dict[str, list[float]] = {
            participant_ids[i]: [float(coords_2d[i, 0]), float(coords_2d[i, 1])]
            for i in range(len(participant_ids))
        }

        # Per-cluster approval rate for each statement
        # approval = fraction of cluster members who agreed (voted +1)
        statement_approval: dict[str, dict[int, float]] = {}
        s_idx = {sid: j for j, sid in enumerate(statement_ids)}

        for sid in statement_ids:
            j = s_idx[sid]
            per_cluster: dict[int, float] = {}
            for cluster in clusters:
                member_indices = [participant_ids.index(pid) for pid in cluster.participant_ids]
                votes = matrix[member_indices, j]
                non_pass = votes[votes != 0]
                if len(non_pass) == 0:
                    per_cluster[cluster.id] = 0.0
                else:
                    per_cluster[cluster.id] = float(np.mean(non_pass > 0))
            statement_approval[str(sid)] = per_cluster

        return ClusteringResult(
            clusters=clusters,
            coordinates_2d=coordinates_2d,
            num_clusters=num_k,
            silhouette_score=sil_score,
            statement_approval_by_cluster=statement_approval,
        )
