"""The GPU lease files of ``gpu-lease.v1`` and ``gpu-lease.ack.v1``.

The live mode of the transcriber holds a lease on the GPU while a
session is on; other processes (the image worker, the batch jobs of the
transcriber) read it and stay off the card while it is valid. This
module only describes the two files and when a lease counts: no I/O.
The JSON schema of the lease is kept in ``docs/platform/CONTRACTS.md``
(section 7); these models are what the transcriber validates against.
"""

from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

LEASE_SCHEMA = "gpu-lease.v1"
ACK_SCHEMA = "gpu-lease.ack.v1"
LIVE_HOLDER = "ragputin-live"
# A lease not renewed for this long is dead; the holder renews it more
# often (see HEARTBEAT_SECONDS), so a crash frees the card within it.
LEASE_TTL_SECONDS = 15.0
HEARTBEAT_SECONDS = 5.0
# How long the holder waits for the other side to say it freed the card.
ACK_TIMEOUT_SECONDS = 20.0

Priority = Literal["live", "batch"]


class LeaseFile(BaseModel):
    """The content of ``lease.json`` (``gpu-lease.v1``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["gpu-lease.v1"] = Field(
        default="gpu-lease.v1", alias="schema"
    )
    holder: str
    priority: Priority
    pid: int
    acquired_at: AwareDatetime
    expires_at: AwareDatetime
    starts_at: Optional[AwareDatetime] = None
    need_mib: int = Field(ge=0)

    def is_valid(self, now: datetime, pid_alive: bool) -> bool:
        """Tell whether the lease counts: not expired and its holder alive."""
        return self.expires_at > now and pid_alive

    def renewed(self, now: datetime) -> "LeaseFile":
        """Return the lease pushed to expire ``LEASE_TTL_SECONDS`` from now."""
        return self.model_copy(
            update={"expires_at": now + timedelta(seconds=LEASE_TTL_SECONDS)}
        )


class AckFile(BaseModel):
    """The content of ``lease.ack.json`` (``gpu-lease.ack.v1``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["gpu-lease.ack.v1"] = Field(
        default="gpu-lease.ack.v1", alias="schema"
    )
    holder: str
    freed: bool
    at: AwareDatetime

    def answers(self, lease: LeaseFile) -> bool:
        """Tell whether this ack says the card is free for ``lease``.

        An ack older than the lease belongs to an earlier lease.
        """
        return self.freed and self.at >= lease.acquired_at


def new_lease(
    holder: str,
    priority: Priority,
    pid: int,
    now: datetime,
    need_mib: int,
    starts_at: Optional[datetime] = None,
) -> LeaseFile:
    """Return a lease acquired ``now`` that expires after the TTL."""
    return LeaseFile(
        holder=holder, priority=priority, pid=pid, acquired_at=now,
        expires_at=now + timedelta(seconds=LEASE_TTL_SECONDS),
        starts_at=starts_at, need_mib=need_mib,
    )
