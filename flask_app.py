# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared Flask application instance.

Constructs the single ``app`` object with template and static folders resolved
for both source and frozen (PyInstaller) runs, so ``app`` and the ``app_*``
blueprint modules can import one instance without a circular dependency.

Main Features:
* Resolves resource paths via ``sys._MEIPASS`` when frozen, else the source tree.
* Sets the global request-body size limit (MAX_CONTENT_LENGTH) to 5 GB for uploads.
"""

import os
import sys

from flask import Flask


def _resource_root():
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(os.path.abspath(__file__))


_RESOURCE_ROOT = _resource_root()
app = Flask(
    __name__,
    template_folder=os.path.join(_RESOURCE_ROOT, 'templates'),
    static_folder=os.path.join(_RESOURCE_ROOT, 'static'),
)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024
