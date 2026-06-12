import os
import time

import pytest

from app_cron import _field_matches, cron_matches_now


FIXED_TS = 1700000000

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


def test_field_matches_star_matches_any_value():
    assert _field_matches('*', 0) is True
    assert _field_matches('*', 59) is True
    assert _field_matches(' * ', 7) is True


def test_field_matches_exact_value():
    assert _field_matches('5', 5) is True
    assert _field_matches('5', 6) is False


def test_field_matches_range_inclusive():
    assert _field_matches('1-5', 1) is True
    assert _field_matches('1-5', 3) is True
    assert _field_matches('1-5', 5) is True


def test_field_matches_range_outside():
    assert _field_matches('1-5', 0) is False
    assert _field_matches('1-5', 6) is False


def test_field_matches_comma_list():
    assert _field_matches('1,3,5', 1) is True
    assert _field_matches('1,3,5', 3) is True
    assert _field_matches('1,3,5', 5) is True
    assert _field_matches('1,3,5', 2) is False
    assert _field_matches('1,3,5', 4) is False


def test_field_matches_malformed_range_returns_false():
    assert _field_matches('1-', 1) is False
    assert _field_matches('1-', 0) is False
    assert _field_matches('-5', 3) is False


def test_field_matches_non_numeric_returns_false():
    assert _field_matches('abc', 1) is False


def test_cron_matches_now_short_expression_returns_false():
    assert cron_matches_now('* * * *', FIXED_TS) is False


@requires_tzset
def test_cron_matches_now_matching_expression(utc_tz):
    assert cron_matches_now('13 22 14 11 *', FIXED_TS) is True


@requires_tzset
def test_cron_matches_now_matching_dow(utc_tz):
    assert cron_matches_now('13 22 * * 2', FIXED_TS) is True


@requires_tzset
def test_cron_matches_now_non_matching_minute(utc_tz):
    assert cron_matches_now('14 22 14 11 *', FIXED_TS) is False


@requires_tzset
def test_cron_matches_now_dom_dow_either_matches(utc_tz):
    assert cron_matches_now('13 22 1 11 2', FIXED_TS) is True
    assert cron_matches_now('13 22 14 11 5', FIXED_TS) is True
    assert cron_matches_now('13 22 1 11 5', FIXED_TS) is False
