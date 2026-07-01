# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Timezone helpers for storing UTC and displaying local time.

Provides the canonical SQL snippet for UTC-now writes and converters that turn
naive/UTC datetimes into the host's local timezone for display, keeping storage
in UTC while presentation is localized.

Main Features:
* ``UTC_NOW_SQL`` constant for consistent UTC timestamps in queries.
* ``to_local`` / ``to_local_str`` convert UTC datetimes to local time and formatted strings.
"""

import datetime


UTC_NOW_SQL = "NOW() AT TIME ZONE 'UTC'"

LOCAL_TZ_FMT = '%Y-%m-%d %H:%M:%S'


def to_local(dt):
    if dt is None or not hasattr(dt, 'astimezone'):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone()


def to_local_str(dt):
    dt = to_local(dt)
    if dt is None:
        return None
    if hasattr(dt, 'strftime'):
        return dt.strftime(LOCAL_TZ_FMT)
    return str(dt)
