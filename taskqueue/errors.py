# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The two things a task may say about its own outcome, and nothing else.

A task tells the queue how it ended by returning or by raising, and the queue
writes the terminal row and decides every retry. These two exceptions are the
only job-specific input to that decision. Everything else a task raises is
retried, because a media-server 502, an LLM timeout, a deadlock or a child the
kernel killed for memory are all things a second attempt can survive.

TaskFailed is for the opposite: input that cannot become valid on a retry - a
func outside the allow-list, a plugin whose module is gone, a server id that no
longer exists, an unsupported media-server type. Never wrap a lost database
connection in it: the worker checks connectivity FIRST, so a wrapped one would
skip the uncharged requeue ladder and be treated as permanent.

TaskCancelled is the cooperative stop. A task raises it from its cancel check
and the queue writes REVOKED, charging no attempt.

This module imports nothing so any task module can import it without touching
the eager-import ceiling.

Main Features:
* TaskFailed: permanent, the queue writes FAIL on the first attempt
* TaskCancelled: the queue writes REVOKED and charges no attempt
"""


class TaskFailed(Exception):
    pass


class TaskCancelled(Exception):
    pass
