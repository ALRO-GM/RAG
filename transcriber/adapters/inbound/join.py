"""The join link ``transcriber serve`` prints, and its terminal QR code.

The access code travels in the URL fragment (``#k=<code>``): a browser
never sends the fragment to a server, so the code reaches no log
(``docs/transcriber/ARCHITECTURE.md`` §5). Through the tunnel the link
points at the public host over https; without it, at the local API.
"""

import io

import segno

LOCAL_HOST = "127.0.0.1"


def join_link(code: str, public_host: str, port: int, tunnel: bool) -> str:
    """Return the link that opens the pages with ``code``.

    ``https://<public_host>/#k=<code>`` with the tunnel,
    ``http://127.0.0.1:<port>/#k=<code>`` without it.
    """
    if tunnel:
        return f"https://{public_host}/#k={code}"
    return f"http://{LOCAL_HOST}:{port}/#k={code}"


def qr_text(link: str) -> str:
    """Return ``link`` as a QR code drawn with block characters."""
    code = segno.make(link, error="m")
    out = io.StringIO()
    code.terminal(out=out, compact=True)
    return out.getvalue()
