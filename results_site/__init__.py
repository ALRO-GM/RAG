"""Static results site: turns the lab's outputs into a browsable website.

The experiments run on the workstation (GPU); this package only reads
what they left in the repository (``experiments/``, ``docs/``,
``knowledge/``) and writes a self-contained static site. The site needs
no server-side code, so the public server only serves files.

Layout:

- ``summaries``: pure functions, raw records -> view models (JSON-ready).
- ``catalog``: which repository files are published and how they group.
- ``builder``: file-system side: read sources, write ``<out>/``.
- ``__main__``: command line (``python -m results_site build``).
"""

SITE_FORMAT_VERSION = 1
