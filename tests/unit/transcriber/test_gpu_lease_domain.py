"""The gpu-lease.v1 and ack models: validation and validity."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from transcriber.domain.gpu_lease import (
    LEASE_TTL_SECONDS, AckFile, LeaseFile, new_lease,
)

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def test_lease_round_trips_with_the_schema_key() -> None:
    lease = new_lease("ragputin-live", "live", 42, NOW, 5000, NOW)
    data = lease.model_dump(mode="json", by_alias=True)
    assert data["schema"] == "gpu-lease.v1"
    assert LeaseFile.model_validate(data) == lease
    assert lease.expires_at == NOW + timedelta(seconds=LEASE_TTL_SECONDS)


def test_lease_rejects_unknown_fields_and_bad_values() -> None:
    good = new_lease("h", "batch", 1, NOW, 0).model_dump(
        mode="json", by_alias=True
    )
    for change in (
        {"extra": 1}, {"priority": "urgent"}, {"need_mib": -1},
        {"schema": "gpu-lease.v2"}, {"expires_at": "2026-01-01T00:00:00"},
    ):
        with pytest.raises(ValidationError):
            LeaseFile.model_validate({**good, **change})


def test_validity_needs_future_expiry_and_live_pid() -> None:
    lease = new_lease("h", "live", 1, NOW, 0)
    assert lease.is_valid(NOW, True)
    assert not lease.is_valid(NOW, False)
    assert not lease.is_valid(lease.expires_at, True)


def test_renewed_pushes_the_expiry() -> None:
    later = NOW + timedelta(seconds=5)
    lease = new_lease("h", "live", 1, NOW, 0).renewed(later)
    assert lease.expires_at == later + timedelta(seconds=LEASE_TTL_SECONDS)
    assert lease.acquired_at == NOW


def test_ack_answers_only_its_own_lease() -> None:
    lease = new_lease("h", "live", 1, NOW, 0)
    fresh = AckFile(holder="img-worker", freed=True, at=NOW)
    stale = AckFile(
        holder="img-worker", freed=True, at=NOW - timedelta(seconds=1)
    )
    busy = AckFile(holder="img-worker", freed=False, at=NOW)
    assert fresh.answers(lease)
    assert not stale.answers(lease)
    assert not busy.answers(lease)
