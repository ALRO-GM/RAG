"""An HTTP opener that never follows redirects.

Requests to the RAG carry its bearer token. ``urllib`` follows a
redirect by sending the same headers to the new address, which could
hand the token to another host, so every request that carries the
token is sent through :data:`OPENER`: a redirect answer comes back as
an ``HTTPError`` with its ``3xx`` status instead of being followed.
"""

import urllib.request
from typing import Any, Optional


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow any redirect."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        """Return ``None``: the ``3xx`` answer is raised as it is."""
        return None


OPENER = urllib.request.build_opener(NoRedirect())
"""Opens requests without following redirects."""
