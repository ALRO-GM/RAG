"""Use case: a person's profile across the stored session assessments."""

from typing import Optional, Sequence

from src.application.skills.ports import AssessmentStore
from src.domain.errors import InvalidInputError
from src.domain.skills.profile import PersonProfile, build_profile


class ProfileService:
    """Join one person's scorecards from several sessions.

    The person is the id the transcriber enrolled them under. Without
    a list of sessions, every stored session where that id appears is
    used; with one, only those (sessions where the person is missing
    are reported as excluded, not silently dropped).
    """

    def __init__(self, store: AssessmentStore, min_situations: int = 3):
        """Use the assessments of ``store``."""
        if min_situations < 1:
            raise InvalidInputError("min_situations must be at least 1.")
        self.store = store
        self.min_situations = min_situations

    def profile(
        self, person_id: str, sessions: Optional[Sequence[str]] = None,
    ) -> PersonProfile:
        """Return the profile of ``person_id``."""
        stored = self.store.all()
        if sessions:
            chosen = [self.store.get(reference) for reference in sessions]
        else:
            chosen = [
                item for item in stored
                if item.scorecard_of_person(person_id) is not None
            ]
        known = {item.session_id for item in stored}
        population = stored + [
            item for item in chosen if item.session_id not in known
        ]
        return build_profile(
            person_id, chosen, self.min_situations, population
        )
