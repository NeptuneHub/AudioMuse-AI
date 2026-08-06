# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The two queue names, and nothing else, importable from anywhere.

These are wire identifiers rather than tunables: the same two strings appear in
the supervisord ``--queue`` arguments, in the native launchers' role dispatch, in
the claim statement's ``queue_name`` column and in the NOTIFY payload workers
filter on. Renaming one without renaming all of them makes the container run two
default workers, and because ``_queue_from_argv`` does not validate, nothing
raises: the high-priority coordinators are simply never claimed and every
analysis sits at "queued" forever.

They deliberately do NOT live in ``config.py``, for two reasons that both bite.
First, they are not tunable - there is no environment override and there cannot
be one, because supervisord's argument and the worker's own idea of its queue
would drift apart. Second, ``config.py`` is layer 0 and the eager import chain is
already at the ``MAX_CHAIN`` ceiling, so giving config a new import breaks
test_import_architecture; this module instead sits BELOW config with no imports
at all.

That zero-import property is what makes it usable from
``taskqueue/worker.py``, which must not touch ``config`` until after it has set
``AUDIOMUSE_ROLE`` and the BLAS thread caps - importing config there loads numpy
and sizes the thread pools against every host CPU before the caps are written.

Main Features:
* ``QUEUE_HIGH`` / ``QUEUE_DEFAULT`` are the only definition of the two names
* ``QUEUE_NAMES`` is the validation list for anything parsing a queue argument
* ``PRIORITY_FRONT`` is the one priority value that jumps the claim order
"""

QUEUE_HIGH = 'high'
QUEUE_DEFAULT = 'default'

QUEUE_NAMES = (QUEUE_HIGH, QUEUE_DEFAULT)

PRIORITY_FRONT = 100

CANCEL_ALL = '*'
