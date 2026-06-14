# sanitization.py
"""Untrusted-input sanitization helpers.

A dependency-free leaf module (standard library + numpy only) so any layer can
clean a value before it reaches the database or a JSON column without pulling in
project modules. Every data-sanitization helper in the project lives here:

- ``sanitize_string_for_db`` -- strip NUL + control characters from a string.
- ``sanitize_db_field``      -- the above plus a length cap and whitespace strip,
                                for a single named DB column.
- ``sanitize_json_for_db``   -- recursively sanitize strings inside dict/list/tuple.
- ``sanitize_for_json``      -- convert numpy scalars/arrays to JSON-native types.
"""
import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_string_for_db(value: Optional[str]) -> Optional[str]:
    """Remove NUL bytes (0x00) and control characters from a string before a DB write.

    PostgreSQL TEXT/VARCHAR columns reject strings containing NUL bytes (0x00),
    which can appear in corrupted metadata from music files. Returns ``None`` for
    ``None``; non-strings are coerced with ``str()``.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    # Remove NUL bytes (0x00)
    value = value.replace('\x00', '')

    # Remove other control characters (0x01-0x1F except tab, newline, carriage return)
    value = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F]', '', value)

    return value


def sanitize_db_field(s, max_length=1000, field_name="field"):
    """Sanitize a single string column value for PostgreSQL insertion.

    Like :func:`sanitize_string_for_db` but also caps the length at
    ``max_length`` and strips surrounding whitespace. ``field_name`` is only used
    in the truncation log message.
    """
    if s is None:
        return None

    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            logger.warning(f"Could not convert {field_name} to string, using empty string")
            return ""

    # Remove NUL byte (0x00) -- PostgreSQL cannot store it.
    s = s.replace('\x00', '')

    # Keep only printable characters plus space/tab/newline.
    s = ''.join(char for char in s if char.isprintable() or char in '\n\t ')

    if len(s) > max_length:
        logger.warning(f"{field_name} truncated from {len(s)} to {max_length} characters")
        s = s[:max_length]

    return s.strip()


def sanitize_json_for_db(value):
    """Recursively sanitize strings inside a JSON-serializable value.

    Applies :func:`sanitize_string_for_db` to every string found inside dicts,
    lists and tuples. Non-string scalars are returned as-is. Use this before
    ``json.dumps(...)`` when the result is going into a Postgres jsonb column.
    """
    if isinstance(value, str):
        return sanitize_string_for_db(value)
    if isinstance(value, dict):
        return {k: sanitize_json_for_db(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_for_db(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_json_for_db(v) for v in value)
    return value


def sanitize_for_json(obj):
    """Recursively convert numpy arrays and numpy numeric types to native Python
    types so the object is JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy numeric types which are not JSON serializable by default
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
