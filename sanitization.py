# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""String and JSON sanitization helpers for safe database and API round-trips.

Strips control characters that Postgres rejects (notably NUL) from text before
persistence, and normalizes numpy scalars/arrays into JSON-serializable values;
used across the data-access layer and API responses.

Main Features:
* ``sanitize_db_field`` removes non-printable characters and truncates to a max length.
* ``sanitize_string_for_db`` / ``sanitize_json_for_db`` strip NUL and control chars from text and nested JSON.
* ``sanitize_for_json`` converts numpy int/float/bool/array types to native Python.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_string_for_db(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.replace('\x00', '')

    value = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F]', '', value)

    return value


def sanitize_db_field(s, max_length=1000, field_name="field"):
    if s is None:
        return None

    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            logger.warning(f"Could not convert {field_name} to string, using empty string")
            return ""

    s = s.replace('\x00', '')

    s = ''.join(char for char in s if char.isprintable() or char in '\n\t ')

    if len(s) > max_length:
        logger.warning(f"{field_name} truncated from {len(s)} to {max_length} characters")
        s = s[:max_length]

    return s.strip()


def sanitize_json_for_db(value):
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
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
