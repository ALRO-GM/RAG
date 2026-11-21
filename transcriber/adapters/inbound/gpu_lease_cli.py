"""``transcriber gpu-lease``: look at, take or drop the GPU lease.

``status`` prints the valid lease (or says the GPU is free); ``acquire``
holds the lease in the foreground, renewing it every 5 s, until
interrupted or ``--duration`` ends, then releases it; ``release``
deletes the lease file (for a stuck one). See ``gpu-lease.v1``.
"""

import argparse
import time
from datetime import datetime
from typing import Callable, Optional

from transcriber.adapters.inbound.cli import Output
from transcriber.application.ports import GpuLease
from transcriber.domain.gpu_lease import ACK_TIMEOUT_SECONDS, LeaseFile

POLL_SECONDS = 0.5


def add_gpu_lease_parser(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    json_flag: Callable[[argparse.ArgumentParser], None],
) -> None:
    """Add the ``gpu-lease`` command and its actions."""
    lease = commands.add_parser(
        "gpu-lease", help="the cooperative GPU lease (gpu-lease.v1)"
    )
    actions = lease.add_subparsers(dest="lease_action", metavar="ACTION")
    actions.required = True
    status = actions.add_parser("status", help="show the valid lease")
    json_flag(status)
    acquire = actions.add_parser(
        "acquire", help="hold the lease until interrupted"
    )
    acquire.add_argument("--need-mib", type=int, required=True,
                         help="GPU memory needed, in MiB")
    acquire.add_argument("--starts-at", help="ISO time the GPU is needed")
    acquire.add_argument("--duration", type=float,
                         help="release after this many seconds")
    acquire.add_argument("--wait-ack", action="store_true",
                         help="wait up to 20 s for the image worker's ack")
    actions.add_parser("release", help="delete the lease file")


def _describe(lease: LeaseFile) -> str:
    """Return one line describing ``lease``."""
    return (
        f"held by {lease.holder} ({lease.priority}, pid {lease.pid}),"
        f" {lease.need_mib} MiB, expires {lease.expires_at.isoformat()}"
    )


class GpuLeaseCommands:
    """Run the ``gpu-lease`` actions over a lease adapter."""

    def __init__(
        self,
        lease: GpuLease,
        output: Output,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Keep the lease adapter, the output and the sleeping function."""
        self._lease = lease
        self._out = output
        self._sleep = sleep

    def run(self, args: argparse.Namespace) -> int:
        """Run the chosen action; return the exit code."""
        if args.lease_action == "status":
            return self.status(args.json)
        if args.lease_action == "acquire":
            return self.acquire(args)
        return self.release()

    def status(self, as_json: bool) -> int:
        """Print the valid lease; exit 0 either way."""
        lease = self._lease.current()
        if as_json:
            self._out.json(
                None if lease is None
                else lease.model_dump(mode="json", by_alias=True)
            )
        else:
            self._out.line(
                "GPU free: no valid lease." if lease is None
                else _describe(lease)
            )
        return 0

    def release(self) -> int:
        """Delete the lease file."""
        if self._lease.clear():
            self._out.line("Lease deleted.")
        else:
            self._out.line("No lease file.")
        return 0

    def acquire(self, args: argparse.Namespace) -> int:
        """Hold the lease until ``--duration`` or Ctrl-C, then release it."""
        starts: Optional[datetime] = (
            datetime.fromisoformat(args.starts_at) if args.starts_at
            else None
        )
        lease = self._lease.acquire(args.need_mib, starts)
        self._out.line(f"Lease acquired: {_describe(lease)}")
        try:
            if args.wait_ack:
                freed = self._wait_ack()
                self._out.line(
                    "Freed by the other side." if freed
                    else "No ack within 20 s: the card may be busy."
                )
            self._hold(args.duration)
        except KeyboardInterrupt:
            pass
        finally:
            self._lease.release()
            self._out.line("Lease released.")
        return 0

    def _wait_ack(self) -> bool:
        """Wait up to the ack timeout for the other side's ack."""
        waited = 0.0
        while waited < ACK_TIMEOUT_SECONDS:
            if self._lease.freed():
                return True
            self._sleep(POLL_SECONDS)
            waited += POLL_SECONDS
        return self._lease.freed()

    def _hold(self, duration: Optional[float]) -> None:
        """Sleep until ``duration`` seconds pass (forever when ``None``)."""
        waited = 0.0
        while duration is None or waited < duration:
            self._sleep(POLL_SECONDS)
            waited += POLL_SECONDS
