# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""User-management page Flask blueprint (users_bp).

Serves the ``/users`` admin HTML page; the page's data is driven client-side by
the user-management API defined alongside authentication, so this blueprint only
renders the shell.

Main Features:
* Single ``/users`` route rendering the users admin template.
"""

import logging

from flask import Blueprint, render_template

logger = logging.getLogger(__name__)

users_bp = Blueprint('users_bp', __name__)


@users_bp.route('/users')
def users_page():
    """
    Users admin page.
    ---
    tags:
      - Users
    summary: HTML page for managing AudioMuse users.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('users.html', title='AudioMuse-AI - Users', active='users')
