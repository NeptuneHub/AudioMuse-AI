"""Shared logging setup for every AudioMuse-AI entry point.

Call ``configure_logging()`` exactly once per process, as early as possible —
before any module that emits log records gets imported. ``logging.basicConfig``
is a no-op if the root logger already has a handler, so calling this helper
multiple times across imports is safe.

Why this exists: workers that don't import ``app`` (the high-priority worker,
the janitor) used to set up logging inline, with formats that drifted from
``app.py``. When one of them forgot to call ``basicConfig`` at all, every
``logger.info(...)`` from task modules fell through to Python's ``lastResort``
handler — silently dropping INFO-level output during long-running jobs.
"""

import logging

LOG_FORMAT = "[%(levelname)s]-[%(asctime)s]-%(message)s"
LOG_DATEFMT = "%d-%m-%Y %H-%M-%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Install the project-wide root logger format. Idempotent."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
