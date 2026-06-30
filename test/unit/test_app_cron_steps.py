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


def test_step_month_anchored_at_one():
    matched = [m for m in range(1, 13) if _field_matches('*/3', m, field_min=1)]
    assert matched == [1, 4, 7, 10]
    assert _field_matches('*/3', 0, field_min=1) is False
    assert _field_matches('*/3', 3, field_min=1) is False
    assert _field_matches('*/3', 6, field_min=1) is False
    assert _field_matches('*/3', 9, field_min=1) is False


def test_step_day_of_month_anchored_at_one():
    matched = [d for d in range(1, 32) if _field_matches('*/2', d, field_min=1)]
    assert matched == list(range(1, 32, 2))
    assert _field_matches('*/2', 2, field_min=1) is False
    assert _field_matches('*/2', 1, field_min=1) is True


def test_step_minute_anchored_at_zero():
    matched = [m for m in range(0, 60) if _field_matches('*/15', m)]
    assert matched == [0, 15, 30, 45]
    assert _field_matches('*/15', 1, field_min=0) is False
    assert _field_matches('*/15', 14, field_min=0) is False


def test_step_hour_anchored_at_zero():
    matched = [h for h in range(0, 24) if _field_matches('*/6', h)]
    assert matched == [0, 6, 12, 18]
    assert _field_matches('*/6', 5, field_min=0) is False
    assert _field_matches('*/6', 23, field_min=0) is False


def test_range_with_step_only_matches_steps_within_range():
    matched = [v for v in range(0, 30) if _field_matches('10-20/5', v)]
    assert matched == [10, 15, 20]
    assert _field_matches('10-20/5', 11) is False
    assert _field_matches('10-20/5', 12) is False
    assert _field_matches('10-20/5', 5) is False
    assert _field_matches('10-20/5', 25) is False


def test_range_with_step_inclusive_upper_bound():
    assert _field_matches('1-10/3', 1) is True
    assert _field_matches('1-10/3', 10) is True
    assert _field_matches('1-10/3', 4) is True
    assert _field_matches('1-10/3', 7) is True
    assert _field_matches('1-10/3', 9) is False


@requires_tzset
def test_dow_seven_is_sunday(utc_tz):
    sunday_ts = time.mktime(time.strptime('2021-08-15 12:00:00', '%Y-%m-%d %H:%M:%S'))
    assert time.localtime(sunday_ts).tm_wday == 6
    assert cron_matches_now('0 12 * * 7', sunday_ts) is True
    assert cron_matches_now('0 12 * * 0', sunday_ts) is True


@requires_tzset
def test_dow_seven_does_not_match_non_sunday(utc_tz):
    monday_ts = time.mktime(time.strptime('2021-08-16 12:00:00', '%Y-%m-%d %H:%M:%S'))
    assert time.localtime(monday_ts).tm_wday == 0
    assert cron_matches_now('0 12 * * 7', monday_ts) is False
    assert cron_matches_now('0 12 * * 0', monday_ts) is False


@requires_tzset
def test_epoch_timestamp_is_evaluated(utc_tz):
    lt = time.localtime(0)
    assert (lt.tm_min, lt.tm_hour, lt.tm_mday, lt.tm_mon) == (0, 0, 1, 1)
    assert cron_matches_now('0 0 1 1 *', 0) is True
    assert cron_matches_now('0 0 * * 4', 0) is True


@requires_tzset
def test_epoch_timestamp_non_matching_minute(utc_tz):
    assert cron_matches_now('30 0 1 1 *', 0) is False
