"""Unit tests for app_cron.py utility functions"""
import pytest
import time
from app_cron import _field_matches, cron_matches_now


class TestFieldMatches:
    """Tests for the _field_matches function"""

    def test_asterisk_matches_any_value(self):
        """Test that '*' matches any value"""
        assert _field_matches('*', 0) is True
        assert _field_matches('*', 15) is True
        assert _field_matches('*', 59) is True
        assert _field_matches('*', 100) is True

    def test_asterisk_with_whitespace(self):
        """Test that '*' with whitespace matches any value"""
        assert _field_matches('  *  ', 5) is True

    def test_single_number_exact_match(self):
        """Test exact match with single number"""
        assert _field_matches('5', 5) is True
        assert _field_matches('5', 6) is False
        assert _field_matches('0', 0) is True
        assert _field_matches('59', 59) is True

    def test_single_number_no_match(self):
        """Test that single number doesn't match different values"""
        assert _field_matches('10', 5) is False
        assert _field_matches('0', 1) is False

    def test_comma_separated_list(self):
        """Test comma-separated list of values"""
        assert _field_matches('1,5,10', 1) is True
        assert _field_matches('1,5,10', 5) is True
        assert _field_matches('1,5,10', 10) is True
        assert _field_matches('1,5,10', 3) is False

    def test_range_within_bounds(self):
        """Test range matching (e.g., '1-5')"""
        assert _field_matches('1-5', 1) is True
        assert _field_matches('1-5', 3) is True
        assert _field_matches('1-5', 5) is True
        assert _field_matches('1-5', 0) is False
        assert _field_matches('1-5', 6) is False

    def test_range_edge_cases(self):
        """Test range edge cases"""
        assert _field_matches('0-0', 0) is True
        assert _field_matches('10-20', 10) is True
        assert _field_matches('10-20', 20) is True
        assert _field_matches('10-20', 15) is True

    def test_mixed_list_and_ranges(self):
        """Test mixing single values and ranges"""
        assert _field_matches('1,5-10,15', 1) is True
        assert _field_matches('1,5-10,15', 7) is True
        assert _field_matches('1,5-10,15', 15) is True
        assert _field_matches('1,5-10,15', 3) is False
        assert _field_matches('1,5-10,15', 12) is False

    def test_whitespace_in_expression(self):
        """Test handling of whitespace in expressions"""
        assert _field_matches('1, 5, 10', 5) is True
        assert _field_matches(' 1 - 5 ', 3) is True

    def test_invalid_number_format_ignored(self):
        """Test that invalid number formats are ignored"""
        assert _field_matches('abc', 5) is False
        assert _field_matches('1,abc,5', 5) is True
        assert _field_matches('1,abc,5', 999) is False

    def test_malformed_range_ignored(self):
        """Test that malformed ranges are ignored"""
        assert _field_matches('1-abc', 1) is False
        assert _field_matches('1-abc,5', 5) is True

    def test_empty_string(self):
        """Test with empty string"""
        assert _field_matches('', 5) is False

    def test_zero_value(self):
        """Test matching with zero"""
        assert _field_matches('0', 0) is True
        assert _field_matches('0-5', 0) is True

    def test_large_values(self):
        """Test with large values"""
        assert _field_matches('100-200', 150) is True
        assert _field_matches('1000', 1000) is True


class TestCronMatchesNow:
    """Tests for the cron_matches_now function"""

    def test_all_asterisks_always_matches(self):
        """Test that '* * * * *' always matches"""
        # Create a specific timestamp: Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* * * * *', ts) is True

    def test_specific_minute_matches(self):
        """Test matching specific minute"""
        # Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('30 * * * *', ts) is True
        assert cron_matches_now('31 * * * *', ts) is False

    def test_specific_hour_matches(self):
        """Test matching specific hour"""
        # Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* 10 * * *', ts) is True
        assert cron_matches_now('* 11 * * *', ts) is False

    def test_specific_minute_and_hour(self):
        """Test matching specific minute and hour"""
        # Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('30 10 * * *', ts) is True
        assert cron_matches_now('30 11 * * *', ts) is False
        assert cron_matches_now('31 10 * * *', ts) is False

    def test_day_of_week_matching(self):
        """Test day of week matching (0=Sunday, 6=Saturday)"""
        # Saturday, June 15, 2024 at 10:30
        sat_ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        # Sunday, June 16, 2024 at 10:30
        sun_ts = time.mktime((2024, 6, 16, 10, 30, 0, 0, 0, -1))
        
        assert cron_matches_now('* * * * 6', sat_ts) is True  # Saturday
        assert cron_matches_now('* * * * 0', sun_ts) is True  # Sunday
        assert cron_matches_now('* * * * 1', sat_ts) is False  # Monday

    def test_range_in_minute_field(self):
        """Test range in minute field"""
        # Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('25-35 * * * *', ts) is True
        assert cron_matches_now('0-20 * * * *', ts) is False

    def test_comma_list_in_hour_field(self):
        """Test comma-separated list in hour field"""
        # Saturday, June 15, 2024 at 10:30
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* 8,10,12 * * *', ts) is True
        assert cron_matches_now('* 8,9,11 * * *', ts) is False

    def test_multiple_days_of_week(self):
        """Test multiple days of week"""
        # Saturday, June 15, 2024 at 10:30
        sat_ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        # Monday, June 17, 2024 at 10:30
        mon_ts = time.mktime((2024, 6, 17, 10, 30, 0, 0, 0, -1))
        
        assert cron_matches_now('* * * * 1,6', sat_ts) is True  # Saturday in list
        assert cron_matches_now('* * * * 1,6', mon_ts) is True  # Monday in list
        assert cron_matches_now('* * * * 2,3', sat_ts) is False

    def test_weekday_range(self):
        """Test range of weekdays"""
        # Wednesday, June 19, 2024 at 10:30
        wed_ts = time.mktime((2024, 6, 19, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* * * * 1-5', wed_ts) is True  # Mon-Fri
        
        # Saturday, June 15, 2024 at 10:30
        sat_ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* * * * 1-5', sat_ts) is False  # Not Mon-Fri

    def test_complex_expression(self):
        """Test complex cron expression"""
        # Monday, June 17, 2024 at 9:30
        ts = time.mktime((2024, 6, 17, 9, 30, 0, 0, 0, -1))
        assert cron_matches_now('30 9 * * 1', ts) is True
        assert cron_matches_now('30 9 * * 2', ts) is False

    def test_midnight(self):
        """Test matching at midnight"""
        # Saturday, June 15, 2024 at midnight
        ts = time.mktime((2024, 6, 15, 0, 0, 0, 0, 0, -1))
        assert cron_matches_now('0 0 * * *', ts) is True
        assert cron_matches_now('0 1 * * *', ts) is False

    def test_end_of_hour(self):
        """Test matching at 59th minute"""
        # Saturday, June 15, 2024 at 23:59
        ts = time.mktime((2024, 6, 15, 23, 59, 0, 0, 0, -1))
        assert cron_matches_now('59 23 * * *', ts) is True
        assert cron_matches_now('59 22 * * *', ts) is False

    def test_insufficient_fields_returns_false(self):
        """Test that expressions with fewer than 5 fields return False"""
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        assert cron_matches_now('* * *', ts) is False
        assert cron_matches_now('30 10', ts) is False
        assert cron_matches_now('*', ts) is False
        assert cron_matches_now('', ts) is False

    def test_uses_current_time_when_no_timestamp(self):
        """Test that function uses current time when no timestamp provided"""
        # This should match since we're using asterisks
        result = cron_matches_now('* * * * *')
        assert result is True

    def test_extra_fields_ignored(self):
        """Test that extra fields beyond 5 are handled"""
        ts = time.mktime((2024, 6, 15, 10, 30, 0, 0, 0, -1))
        # Expression has 7 fields, only first 5 should be used
        assert cron_matches_now('30 10 * * * extra fields', ts) is True

    def test_every_5_minutes(self):
        """Test pattern for every 5 minutes (simulated with list)"""
        # This tests if our function can handle lists representing intervals
        ts_00 = time.mktime((2024, 6, 15, 10, 0, 0, 0, 0, -1))
        ts_05 = time.mktime((2024, 6, 15, 10, 5, 0, 0, 0, -1))
        ts_03 = time.mktime((2024, 6, 15, 10, 3, 0, 0, 0, -1))
        
        assert cron_matches_now('0,5,10,15,20,25,30,35,40,45,50,55 * * * *', ts_00) is True
        assert cron_matches_now('0,5,10,15,20,25,30,35,40,45,50,55 * * * *', ts_05) is True
        assert cron_matches_now('0,5,10,15,20,25,30,35,40,45,50,55 * * * *', ts_03) is False

    def test_business_hours(self):
        """Test pattern for business hours (9-17, Mon-Fri)"""
        # Monday, June 17, 2024 at 10:00
        mon_work = time.mktime((2024, 6, 17, 10, 0, 0, 0, 0, -1))
        # Saturday, June 15, 2024 at 10:00
        sat_work = time.mktime((2024, 6, 15, 10, 0, 0, 0, 0, -1))
        # Monday, June 17, 2024 at 18:00
        mon_evening = time.mktime((2024, 6, 17, 18, 0, 0, 0, 0, -1))
        
        assert cron_matches_now('0 9-17 * * 1-5', mon_work) is True
        assert cron_matches_now('0 9-17 * * 1-5', sat_work) is False
        assert cron_matches_now('0 9-17 * * 1-5', mon_evening) is False
