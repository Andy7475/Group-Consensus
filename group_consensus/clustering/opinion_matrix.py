"""Tracks participant votes on statements and builds the opinion matrix.

The opinion matrix is the core data structure for Polis-style analysis:
  rows = participants
  columns = statements
  values = +1 (agree), -1 (disagree), 0 (pass or no vote)

Missing votes are treated as 0 for clustering purposes but are tracked
separately so routing logic can prioritise unseen statements.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd

from group_consensus.models.types import Statement, Vote, VoteValue


class OpinionMatrix:
    """Manages the participant × statement vote matrix."""

    def __init__(self) -> None:
        self._votes: dict[tuple[str, UUID], int] = {}
        self._participant_ids: list[str] = []
        self._statement_ids: list[UUID] = []

    def add_vote(self, vote: Vote) -> None:
        pid = vote.participant_id
        sid = vote.statement_id
        if pid not in self._participant_ids:
            self._participant_ids.append(pid)
        if sid not in self._statement_ids:
            self._statement_ids.append(sid)
        self._votes[(pid, sid)] = vote.value.value

    def add_statement(self, statement: Statement) -> None:
        if statement.id not in self._statement_ids:
            self._statement_ids.append(statement.id)

    def add_participant(self, participant_id: str) -> None:
        if participant_id not in self._participant_ids:
            self._participant_ids.append(participant_id)

    @property
    def participant_ids(self) -> list[str]:
        return list(self._participant_ids)

    @property
    def statement_ids(self) -> list[UUID]:
        return list(self._statement_ids)

    def to_numpy(self) -> np.ndarray:
        """Return the matrix as a float32 numpy array (participants × statements)."""
        n_p = len(self._participant_ids)
        n_s = len(self._statement_ids)
        matrix = np.zeros((n_p, n_s), dtype=np.float32)
        p_idx = {pid: i for i, pid in enumerate(self._participant_ids)}
        s_idx = {sid: i for i, sid in enumerate(self._statement_ids)}
        for (pid, sid), val in self._votes.items():
            matrix[p_idx[pid], s_idx[sid]] = val
        return matrix

    def to_dataframe(self) -> pd.DataFrame:
        matrix = self.to_numpy()
        return pd.DataFrame(
            matrix,
            index=self._participant_ids,
            columns=[str(sid) for sid in self._statement_ids],
        )

    def statement_approval_rates(self) -> dict[UUID, float]:
        """Fraction of non-pass votes that are agreements, per statement."""
        matrix = self.to_numpy()
        rates: dict[UUID, float] = {}
        for j, sid in enumerate(self._statement_ids):
            col = matrix[:, j]
            non_pass = col[col != 0]
            if len(non_pass) == 0:
                rates[sid] = 0.0
            else:
                rates[sid] = float(np.mean(non_pass == 1))
        return rates

    def next_statements_for_participant(
        self, participant_id: str, n: int = 5
    ) -> list[UUID]:
        """
        Return up to n statement IDs the participant hasn't voted on yet,
        prioritising high-variance (divisive) statements — same routing
        logic as Polis.
        """
        voted = {sid for (pid, sid), _ in self._votes.items() if pid == participant_id}
        unseen = [sid for sid in self._statement_ids if sid not in voted]

        if not unseen:
            return []

        matrix = self.to_numpy()
        s_idx = {sid: i for i, sid in enumerate(self._statement_ids)}

        # Score by variance across all votes on each statement
        def variance_score(sid: UUID) -> float:
            col = matrix[:, s_idx[sid]]
            voted_vals = col[col != 0]
            return float(np.var(voted_vals)) if len(voted_vals) > 0 else 0.0

        return sorted(unseen, key=variance_score, reverse=True)[:n]

    def __len__(self) -> int:
        return len(self._votes)
