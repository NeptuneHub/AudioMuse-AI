# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The two queue names, and nothing else, importable from anywhere.

These are wire identifiers rather than tunables: the same two strings appear in
the supervisord --queue arguments, the native launchers' role dispatch, the
claim statement's queue_name column and the NOTIFY payload workers filter on.
Renaming one without renaming all of them makes the container run two default
workers and the high-priority coordinators are never claimed.

They deliberately do NOT live in config.py: they are not tunable (no
environment override, so supervisord's argument and the worker's idea of its
queue can never drift), and config.py is layer 0 already at the MAX_CHAIN import
ceiling. This module sits below config with no imports at all, which is what
makes it usable from taskqueue/worker.py before config (and numpy) are touched.

Main Features:
* QUEUE_HIGH / QUEUE_DEFAULT are the only definition of the two names
* QUEUE_NAMES is the validation list for anything parsing a queue argument
* PRIORITY_FRONT is the one priority value that jumps the claim order
"""

QUEUE_HIGH = 'high'
QUEUE_DEFAULT = 'default'

QUEUE_NAMES = (QUEUE_HIGH, QUEUE_DEFAULT)

PRIORITY_FRONT = 100

CANCEL_ALL = '*'
