"""K-means clustering on UMAP-reduced opinion data.

- Reduce to 2D with UMAP (cosine metric) — suits {-1, 0, +1} vote vectors
- Run K-means on the 2D coordinates so clusters match what's visible in the plot
- Select k by silhouette score

The output is a ClusteringResult containing cluster assignments, 2D coordinates,
and per-cluster statement approval rates.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

from group_consensus.clustering.opinion_matrix import OpinionMatrix
from group_consensus.models.types import Cluster, ClusteringResult


def _best_k(matrix: np.ndarray, max_k: int) -> tuple[int, float, np.ndarray]:
    """Find the k with the best silhouette score. Returns (k, score, labels)."""
    if matrix.shape[0] < 4:
        labels = np.zeros(matrix.shape[0], dtype=int)
        return 1, 0.0, labels

    best_k = 2
    best_score = -1.0
    best_labels = np.zeros(matrix.shape[0], dtype=int)

    for k in range(2, min(max_k + 1, matrix.shape[0])):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)
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
        participant_ids = opinion_matrix.participant_ids
        statement_ids = opinion_matrix.statement_ids

        if not participant_ids or not statement_ids:
            raise ValueError("Opinion matrix is empty — add participants and votes first.")

        matrix = opinion_matrix.to_numpy()

        # UMAP to 2D first — cosine metric, then K-means on these coordinates
        # so that clustering and the visual plot are in the same space.
        n_neighbors = min(matrix.shape[0] - 1, 10)
        if matrix.shape[0] >= 4:
            reducer = UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
            )
            coords_2d = reducer.fit_transform(matrix)
        else:
            coords_2d = np.zeros((matrix.shape[0], 2), dtype=np.float32)

        num_k, sil_score, labels = _best_k(coords_2d, self.max_clusters)

        # Build Cluster objects; centroid is in UMAP 2D space
        clusters: list[Cluster] = []
        for k in range(num_k):
            member_indices = np.where(labels == k)[0]
            member_ids = [participant_ids[i] for i in member_indices]
            centroid = coords_2d[member_indices].mean(axis=0).tolist()
            clusters.append(Cluster(id=k, participant_ids=member_ids, centroid=centroid))

        coordinates_2d: dict[str, list[float]] = {
            participant_ids[i]: [float(coords_2d[i, 0]), float(coords_2d[i, 1])]
            for i in range(len(participant_ids))
        }

        # Per-cluster approval from the original vote matrix
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
