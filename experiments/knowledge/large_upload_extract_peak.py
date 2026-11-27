"""Extract one PDF with the upload readers and print time and peak RSS.

Docling on the CPU, no captions, in a process of its own: used to tell
the models' memory from the file's (#282, docs/KNOWLEDGE.md §4.2).

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \\
    python experiments/knowledge/large_upload_extract_peak.py FILE.pdf
"""

import json
import resource
import sys
import time
from pathlib import Path

from src.adapters.outbound.docling_pages import DoclingPages
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.application.knowledge.ports import StoredFile


def main() -> None:
    """Extract ``sys.argv[1]``; print one JSON line."""
    path = Path(sys.argv[1])
    size = path.stat().st_size
    started = time.perf_counter()
    found = DocumentExtractors(DoclingPages(None)).extract(
        StoredFile(str(path), size), "pdf"
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({
        "file_mb": round(size / 1e6, 1), "pages": found.pages,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_mib": round(peak),
    }))


if __name__ == "__main__":
    main()
