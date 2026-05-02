"""Champion tracking service for promoted local models.
Exists to keep active model selection and rollback metadata outside raw run state.
Connects successful challengers to one current local champion per scope.
"""

from __future__ import annotations

from pathlib import Path

from echozero.foundry.domain import ChampionModelRecord, ModelCandidateRecord
from echozero.foundry.persistence import ChampionModelRepository


class ChampionService:
    """Track the current promoted challenger for one scope."""

    def __init__(
        self,
        root: Path,
        *,
        repository: ChampionModelRepository | None = None,
    ) -> None:
        self._root = root
        self._repository = repository or ChampionModelRepository(root)

    def promote_candidate(
        self,
        candidate: ModelCandidateRecord,
        *,
        scope: str = "local.default",
        notes: dict[str, object] | None = None,
    ) -> ChampionModelRecord:
        """Promote one candidate into the active champion slot."""
        if not candidate.artifact_id:
            raise ValueError("Cannot promote a candidate without an artifact")
        previous = self._repository.get(scope)
        champion = ChampionModelRecord(
            scope=scope,
            candidate_id=candidate.id,
            artifact_id=candidate.artifact_id,
            previous_candidate_id=previous.candidate_id if previous else None,
            notes=dict(notes or {}),
        )
        return self._repository.save(champion)

    def get_champion(self, scope: str = "local.default") -> ChampionModelRecord | None:
        """Return the current champion for one scope."""
        return self._repository.get(scope)
