# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""WSGI middleware that strips a duplicated reverse-proxy path prefix.

Wraps the Flask app so that when a proxy sets ``SCRIPT_NAME`` and also leaves
the same prefix in ``PATH_INFO``, the duplicate is removed before routing,
keeping URL matching correct behind subpath reverse proxies.

Main Features:
* Removes a leading ``SCRIPT_NAME`` prefix duplicated in ``PATH_INFO``.
* Leaves requests untouched when no prefix is present.
"""


class StripDuplicatedScriptName:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = environ.get('SCRIPT_NAME', '').rstrip('/')
        if prefix:
            path_info = environ.get('PATH_INFO', '')
            if path_info == prefix or path_info.startswith(prefix + '/'):
                environ['PATH_INFO'] = path_info[len(prefix) :] or '/'
        return self.app(environ, start_response)
