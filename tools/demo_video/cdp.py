"""A small Chrome DevTools Protocol client: commands, events, screencast.

Only what the recording needs: send a command and wait for its answer,
evaluate JavaScript, click at a point, and keep the screencast frames
(JPEG files on disk with Chrome's own capture time) while the page
plays. Frames are acknowledged as soon as they are written, so Chrome
keeps sending them at its own pace.
"""

import asyncio
import base64
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from websockets.asyncio.client import ClientConnection, connect


class CdpError(RuntimeError):
    """A DevTools command answered with an error."""


class CdpPage:
    """One page target, driven over its DevTools WebSocket."""

    def __init__(self, socket: ClientConnection) -> None:
        """Wrap an open connection to a page target."""
        self.socket = socket
        self.ids = itertools.count(1)
        self.pending: Dict[int, "asyncio.Future[Dict[str, Any]]"] = {}
        self.frames: List[Tuple[float, Path]] = []
        self.frame_dir: Optional[Path] = None
        self.reader: Optional["asyncio.Task[None]"] = None

    @classmethod
    async def open(cls, url: str) -> "CdpPage":
        """Connect to the page's ``webSocketDebuggerUrl``."""
        socket = await connect(url, max_size=64_000_000, ping_interval=None)
        page = cls(socket)
        page.reader = asyncio.create_task(page._read())
        return page

    async def close(self) -> None:
        """Close the connection."""
        if self.reader is not None:
            self.reader.cancel()
        await self.socket.close()

    async def _read(self) -> None:
        """Route answers to their callers and keep screencast frames."""
        async for raw in self.socket:
            message = json.loads(raw)
            ident = message.get("id")
            if ident is not None and ident in self.pending:
                self.pending.pop(ident).set_result(message)
            elif message.get("method") == "Page.screencastFrame":
                await self._keep_frame(message["params"])

    async def _keep_frame(self, params: Dict[str, Any]) -> None:
        """Write one frame to disk and acknowledge it."""
        if self.frame_dir is not None:
            path = self.frame_dir / f"{len(self.frames):06d}.jpg"
            path.write_bytes(base64.b64decode(params["data"]))
            stamp = float(params["metadata"]["timestamp"])
            self.frames.append((stamp, path))
        ident = next(self.ids)
        await self.socket.send(json.dumps({
            "id": ident, "method": "Page.screencastFrameAck",
            "params": {"sessionId": params["sessionId"]},
        }))

    async def send(self, method: str, **params: Any) -> Dict[str, Any]:
        """Send one command and return its result."""
        ident = next(self.ids)
        future: "asyncio.Future[Dict[str, Any]]" = (
            asyncio.get_running_loop().create_future()
        )
        self.pending[ident] = future
        await self.socket.send(json.dumps(
            {"id": ident, "method": method, "params": params}
        ))
        answer = await future
        if "error" in answer:
            raise CdpError(f"{method}: {answer['error']}")
        result: Dict[str, Any] = answer.get("result", {})
        return result

    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in the page and return its JSON value."""
        result = await self.send(
            "Runtime.evaluate", expression=expression,
            awaitPromise=True, returnByValue=True,
        )
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            text = details.get("exception", {}).get("description", details)
            raise CdpError(f"JavaScript failed: {text}")
        return result.get("result", {}).get("value")

    async def click(self, x: float, y: float) -> None:
        """Press and release the left button at ``(x, y)`` (CSS pixels)."""
        for kind in ("mousePressed", "mouseReleased"):
            await self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y,
                            button="left", clickCount=1)

    async def start_screencast(self, frame_dir: Path, width: int,
                               height: int, quality: int) -> None:
        """Start keeping JPEG frames of at most ``width`` x ``height``."""
        frame_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = frame_dir
        await self.send("Page.startScreencast", format="jpeg",
                        quality=quality, maxWidth=width, maxHeight=height,
                        everyNthFrame=1)

    async def stop_screencast(self) -> None:
        """Stop the screencast (frames already kept stay)."""
        await self.send("Page.stopScreencast")
        self.frame_dir = None
