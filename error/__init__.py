# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Public interface for the centralized error handling package.

Re-exports the error registry from ``error_dictionary`` and the classification
and formatting helpers from ``error_manager`` so callers import structured
error codes, classes, and messages from a single ``error`` namespace.

Main Features:
* Flattens the dictionary and manager symbols into one import surface.
* Defines ``__all__`` to pin the package's stable public API.
"""

from error.error_dictionary import (
    ERROR_REGISTRY,
    UNKNOWN_ERROR_CODE,
    get_error_class,
    get_default_message,
)
from error.error_manager import (
    AudioMuseError,
    ErrorManager,
    build,
    record,
    classify,
    from_exception,
    http_status_for_code,
)

__all__ = [
    "ERROR_REGISTRY",
    "UNKNOWN_ERROR_CODE",
    "get_error_class",
    "get_default_message",
    "AudioMuseError",
    "ErrorManager",
    "build",
    "record",
    "classify",
    "from_exception",
    "http_status_for_code",
]
