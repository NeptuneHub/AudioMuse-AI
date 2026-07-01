# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Timezone conversion of stored timestamps into local display time.

Covers tz_helper converting naive and aware datetimes to the configured local
zone, including DST handling and non-datetime passthrough.

Main Features:
* None, string and int inputs pass through unchanged
* Naive timestamps are treated as UTC, applying DST for summer dates
* Aware UTC input matches the naive-UTC result
* The string formatter returns None for None and formats naive winter times
"""

import datetime
import time

import pytest

import tz_helper

pytestmark = pytest.mark.skipif(
    not hasattr(time, 'tzset'),
    reason="time.tzset() not available on this platform",
)


@pytest.fixture
def new_york_tz(monkeypatch):
    monkeypatch.setenv('TZ', 'America/New_York')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_to_local_none_passes_through(new_york_tz):
    assert tz_helper.to_local(None) is None


def test_to_local_string_passes_through(new_york_tz):
    assert tz_helper.to_local('x') == 'x'


def test_to_local_int_passes_through(new_york_tz):
    assert tz_helper.to_local(5) == 5


def test_to_local_naive_winter_treated_as_utc(new_york_tz):
    result = tz_helper.to_local(datetime.datetime(2026, 1, 15, 12, 0))
    assert result.hour == 7
    assert result.utcoffset() == datetime.timedelta(hours=-5)


def test_to_local_naive_summer_applies_dst(new_york_tz):
    result = tz_helper.to_local(datetime.datetime(2026, 7, 15, 12, 0))
    assert result.hour == 8
    assert result.utcoffset() == datetime.timedelta(hours=-4)


def test_to_local_aware_utc_matches_naive(new_york_tz):
    naive_result = tz_helper.to_local(datetime.datetime(2026, 1, 15, 12, 0))
    aware_result = tz_helper.to_local(
        datetime.datetime(2026, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
    )
    assert aware_result == naive_result
    assert aware_result.hour == 7


def test_to_local_str_none_returns_none(new_york_tz):
    assert tz_helper.to_local_str(None) is None


def test_to_local_str_formats_naive_winter(new_york_tz):
    result = tz_helper.to_local_str(datetime.datetime(2026, 1, 15, 12, 0))
    assert result == '2026-01-15 07:00:00'
