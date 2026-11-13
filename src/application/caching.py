"""Cache key derivation shared by the use cases."""

import hashlib
import json


def cache_key(*parts: object) -> str:
    """Derive a stable key from JSON-serialisable parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
