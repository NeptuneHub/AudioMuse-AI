"""Tests for the app_cron.py cron matcher step/range refinements.

Covers:
- '*/N' anchored at the field minimum (month/dom start at 1, minute/hour at 0)
- range-with-step 'a-b/N'
- day-of-week 7 treated as Sunday (== 0)
- a timestamp of 0 (the epoch) evaluated correctly, not skipped as falsy

These exercise the real _field_matches / cron_matches_now functions directly.
Cases here intentionally do NOT overlap test_app_cron_parsing.py.
"""
import os
import time

import pytest

from app_cron import _field_matches, cron_matches_now


requires_tzset = pytest.mark.skipif(
    not hasattr(time, 'tzset'), reason='time.tzset not available on this platform'
)


@pytest.fixture
def utc_tz():
    old = os.environ.get('TZ')
    os.environ['TZ'] = 'UTC'
    time.tzset()
    yield
    if old is None:
        os.environ.pop('TZ', None)
    else:
        os.environ['TZ'] = old
    time.tzset()


# --- (a) '*/N' anchored at the field minimum --------------------------------

def test_step_month_anchored_at_one():
    # month '*/3' -> 1,4,7,10 (field_min=1), NOT 0,3,6,9
    matched = [m for m in range(1, 13) if _field_matches('*/3', m, field_min=1)]
    assert matched == [1, 4, 7, 10]
    assert _field_matches('*/3', 0, field_min=1) is False
    assert _field_matches('*/3', 3, field_min=1) is False
    assert _field_matches('*/3', 6, field_min=1) is False
    assert _field_matches('*/3', 9, field_min=1) is False


def test_step_day_of_month_anchored_at_one():
    # dom '*/2' -> odd days 1,3,5,7,... (field_min=1)
    matched = [d for d in range(1, 32) if _field_matches('*/2', d, field_min=1)]
    assert matched == list(range(1, 32, 2))
    assert _field_matches('*/2', 2, field_min=1) is False
    assert _field_matches('*/2', 1, field_min=1) is True


def test_step_minute_anchored_at_zero():
    # minute '*/15' -> 0,15,30,45 (field_min=0)
    matched = [m for m in range(0, 60) if _field_matches('*/15', m)]
    assert matched == [0, 15, 30, 45]
    assert _field_matches('*/15', 1, field_min=0) is False
    assert _field_matches('*/15', 14, field_min=0) is False


def test_step_hour_anchored_at_zero():
    # hour '*/6' -> 0,6,12,18 (field_min=0)
    matched = [h for h in range(0, 24) if _field_matches('*/6', h)]
    assert matched == [0, 6, 12, 18]
    assert _field_matches('*/6', 5, field_min=0) is False
    assert _field_matches('*/6', 23, field_min=0) is False


# --- (b) range-with-step 'a-b/N' --------------------------------------------

def test_range_with_step_only_matches_steps_within_range():
    # '10-20/5' -> 10,15,20 and nothing else in 0..29
    matched = [v for v in range(0, 30) if _field_matches('10-20/5', v)]
    assert matched == [10, 15, 20]
    # boundaries / off-step / out-of-range all excluded
    assert _field_matches('10-20/5', 11) is False
    assert _field_matches('10-20/5', 12) is False
    assert _field_matches('10-20/5', 5) is False
    assert _field_matches('10-20/5', 25) is False


def test_range_with_step_inclusive_upper_bound():
    # upper bound matches only when on-step from lo
    assert _field_matches('1-10/3', 1) is True
    assert _field_matches('1-10/3', 10) is True
    assert _field_matches('1-10/3', 4) is True
    assert _field_matches('1-10/3', 7) is True
    assert _field_matches('1-10/3', 9) is False


# --- (c) day-of-week 7 treated as Sunday (== 0) -----------------------------

@requires_tzset
def test_dow_seven_is_sunday(utc_tz):
    # 2021-08-15 12:00:00 UTC is a Sunday.
    sunday_ts = time.mktime(time.strptime(
        '2021-08-15 12:00:00', '%Y-%m-%d %H:%M:%S'))
    assert time.localtime(sunday_ts).tm_wday == 6  # Python Sunday
    # dow '7' should match a Sunday (cron 7 == 0 == Sunday)
    assert cron_matches_now('0 12 * * 7', sunday_ts) is True
    # dow '0' should also match the same Sunday
    assert cron_matches_now('0 12 * * 0', sunday_ts) is True


@requires_tzset
def test_dow_seven_does_not_match_non_sunday(utc_tz):
    # 2021-08-16 12:00:00 UTC is a Monday.
    monday_ts = time.mktime(time.strptime(
        '2021-08-16 12:00:00', '%Y-%m-%d %H:%M:%S'))
    assert time.localtime(monday_ts).tm_wday == 0  # Python Monday
    assert cron_matches_now('0 12 * * 7', monday_ts) is False
    assert cron_matches_now('0 12 * * 0', monday_ts) is False


# --- (d) the epoch (ts == 0) is evaluated, not skipped as falsy -------------

@requires_tzset
def test_epoch_timestamp_is_evaluated(utc_tz):
    # localtime(0) under UTC = 1970-01-01 00:00:00, a Thursday (py_dow 4).
    lt = time.localtime(0)
    assert (lt.tm_min, lt.tm_hour, lt.tm_mday, lt.tm_mon) == (0, 0, 1, 1)
    # Matching expr for the epoch: Jan 1, 00:00.
    assert cron_matches_now('0 0 1 1 *', 0) is True
    # Thursday at midnight also matches via dow (cron Thursday == 4).
    assert cron_matches_now('0 0 * * 4', 0) is True


@requires_tzset
def test_epoch_timestamp_non_matching_minute(utc_tz):
    # ts=0 must still be honored (not treated as "use now"): a non-matching
    # minute returns False rather than silently matching the current time.
    assert cron_matches_now('30 0 1 1 *', 0) is False
