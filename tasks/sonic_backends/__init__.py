# tasks/sonic_backends/__init__.py
"""Sonic-analysis backend registry.

Backend modules are imported lazily on first ``get_backend()`` call so
that simply importing ``tasks.sonic_backends`` does not pull in optional
ML dependencies (transformers, torch) that the default ``musicnn``
backend does not need.
"""

from .base import SonicBackend, available_backends, get_backend, register

_LOADED = False


def _ensure_loaded() -> None:
    """Import every shipped backend module exactly once.

    Each module registers itself by instantiating its backend class at
    import time. Imports that fail (e.g. MERT when ``transformers`` is
    not installed) are logged at warning level and the rest still
    register — selecting a missing backend then raises a clear error in
    ``get_backend()``.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True  # guard against re-entry on import errors

    import logging
    logger = logging.getLogger(__name__)

    # musicnn must always be importable — it is the default backend and
    # has no optional deps.
    from . import musicnn  # noqa: F401

    for optional in ("mert",):
        try:
            __import__(f"{__name__}.{optional}")
        except Exception as e:  # noqa: BLE001
            logger.info(
                "Sonic backend '%s' not available: %s. "
                "Install its extras to enable.",
                optional, e,
            )


__all__ = ["SonicBackend", "available_backends", "get_backend", "register"]
