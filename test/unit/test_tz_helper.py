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
