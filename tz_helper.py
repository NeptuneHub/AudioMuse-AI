"""Timezone helpers for storing and displaying datetimes consistently
(see issue #499).

Convention used across the app:

  * Naive ``TIMESTAMP`` columns store UTC wall-clock. INSERTs that fill
    such columns must use ``UTC_NOW_SQL`` instead of bare ``NOW()`` /
    relying on a ``DEFAULT CURRENT_TIMESTAMP``, so the stored bytes are
    UTC regardless of the Postgres session timezone.

  * On read, datetimes are passed through ``to_local()`` to convert into
    the Flask container's local timezone (driven by the ``TZ`` env var)
    before serialization. The frontend then renders them as-is.
"""
import datetime


#: SQL fragment that resolves to "now" expressed as a UTC wall-clock,
#: independent of the Postgres session timezone. Drop into INSERT/UPDATE
#: in place of ``NOW()`` for any naive ``TIMESTAMP`` column we want to
#: render later via :func:`to_local`.
UTC_NOW_SQL = "NOW() AT TIME ZONE 'UTC'"

#: ``strftime`` format used for every user-visible timestamp the API
#: returns. Centralised so the wire format stays identical across every
#: endpoint and every frontend renderer can stay dumb (raw concat).
LOCAL_TZ_FMT = '%Y-%m-%d %H:%M:%S'


def to_local(dt):
    """Return ``dt`` converted to the local timezone.

    Naive inputs are assumed to be UTC (matching what ``UTC_NOW_SQL``
    writes). Tz-aware inputs are re-converted to local. ``None`` and
    non-datetime values pass through unchanged so the helper can be
    used inline at serialization time without extra guards.
    """
    if dt is None or not hasattr(dt, 'astimezone'):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone()


def to_local_str(dt):
    """Convert ``dt`` to local time and format with ``LOCAL_TZ_FMT``.

    Returns ``None`` for ``None``. Non-datetime values are coerced via
    ``str()`` to preserve the prior defensive behavior of callers.
    """
    dt = to_local(dt)
    if dt is None:
        return None
    if hasattr(dt, 'strftime'):
        return dt.strftime(LOCAL_TZ_FMT)
    return str(dt)
