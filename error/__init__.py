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
* Flattens the dictionary and manager symbols into one import surface, resolved
  on first access (PEP 562) so importing any ``error.*`` module never loads the
  whole package eagerly: every route module imports a code constant, and an
  eager re-export here added two levels to each of their import chains.
* Defines ``__all__`` to pin the package's stable public API.
"""

import importlib

_EXPORTS = {
    "ERROR_REGISTRY": "error.error_dictionary",
    "UNKNOWN_ERROR_CODE": "error.error_dictionary",
    "get_error_class": "error.error_dictionary",
    "get_default_message": "error.error_dictionary",
    "AudioMuseError": "error.error_manager",
    "build": "error.error_manager",
    "record": "error.error_manager",
    "classify": "error.error_manager",
    "from_exception": "error.error_manager",
    "http_status_for_code": "error.error_manager",
    "is_out_of_memory": "error.error_manager",
    "is_model_out_of_memory": "error.error_manager",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'error' has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
