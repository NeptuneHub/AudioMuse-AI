# tasks/sonic_backends/__init__.py
"""Sonic-analysis backend registry.

Backend modules are imported lazily on first ``get_backend()`` call so
that simply importing ``tasks.sonic_backends`` does not pull in optional
ML dependencies (transformers, torch) that the default ``musicnn``
backend does not need.
"""

from .base import SonicBackend, available_backends, get_backend, register


def active_backend_name() -> str:
    """Return the currently configured backend name.

    Read lazily from :mod:`config` so reloading the module (e.g. during
    tests that monkeypatch ``SONIC_BACKEND``) picks up the new value.
    """
    from config import SONIC_BACKEND
    return SONIC_BACKEND


def voyager_index_name(backend: str | None = None) -> str:
    """Per-backend Voyager primary key in ``voyager_index_data``.

    Always namespaced: legacy bare-named ``music_library`` rows are
    migrated to ``music_library_musicnn`` at ``init_db`` time so every
    row has an unambiguous backend tag. When ``backend`` is None, the
    active backend is used.
    """
    from config import INDEX_NAME
    name = backend or active_backend_name()
    return f"{INDEX_NAME}_{name}"

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


__all__ = [
    "SonicBackend", "active_backend_name", "available_backends",
    "get_backend", "register", "voyager_index_name",
]
