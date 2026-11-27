"""The transcriber's real app for the large-upload run (#282).

``create_app`` over ``start_container`` (as the service starts), with
fake speech models: the ``/api/knowledge`` forwarding and the
``/internal/knowledge-media`` hand-over are the real ones. ``WORK``
holds ``token`` (the RAG's) and ``code`` (the access code):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> \\
    python experiments/knowledge/large_upload_transcriber.py WORK PORT RAG_PORT

How the whole run goes: ``large_upload_client.py``.
"""

import os
import sys
from pathlib import Path

import uvicorn

from transcriber.adapters.inbound.http_api import create_app
from transcriber.bootstrap import start_container
from transcriber.config import TranscriberSettings


def main() -> None:
    """Serve the app until stopped."""
    work, port, rag_port = Path(sys.argv[1]), int(sys.argv[2]), int(
        sys.argv[3]
    )
    os.environ["TRANSCRIBER_PUBLISH_TOKEN"] = (
        (work / "token").read_text().strip()
    )
    settings = TranscriberSettings.model_validate({
        "data_dir": str(work / "transcriber"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": f"http://127.0.0.1:{rag_port}"},
    })
    container = start_container(settings)
    print("pid", os.getpid(), flush=True)
    uvicorn.run(
        create_app(container, (work / "code").read_text().strip()),
        host="127.0.0.1", port=port, log_level="warning",
    )


if __name__ == "__main__":
    main()
