# Shared flask-limiter instance.
#
# Defined in its own module (importing nothing from app.py / app_auth.py) so
# both the app factory and the auth routes can attach limits without a circular
# import. Enforcement only happens once limiter.init_app(app) runs in app.py;
# unit tests that build a bare Flask app and never call it are unaffected.
#
# flask-limiter is treated as optional: if it is not installed (e.g. a slim
# build variant), rate limiting degrades to a no-op rather than crashing the
# app at import time.

import logging

import config

logger = logging.getLogger(__name__)


class _NoopLimiter:
    # Stand-in used when flask-limiter is unavailable. limit() returns an
    # identity decorator and init_app() does nothing, so callers are unchanged.
    def limit(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def init_app(self, app):
        return None


try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    # default_limits is empty: only explicitly decorated endpoints (login, user
    # administration) are limited, so normal high-frequency polling is never
    # throttled. storage falls back to in-process memory when Redis is
    # unreachable; swallow_errors keeps a storage hiccup from 500-ing a request
    # (it fails open rather than locking users out).
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[],
        storage_uri=config.RATE_LIMIT_STORAGE_URI or "memory://",
        strategy="fixed-window",
        swallow_errors=True,
        enabled=config.RATE_LIMIT_ENABLED,
    )
except ImportError:
    logger.warning("flask-limiter not installed; API rate limiting is disabled.")
    limiter = _NoopLimiter()  # type: ignore[assignment]
