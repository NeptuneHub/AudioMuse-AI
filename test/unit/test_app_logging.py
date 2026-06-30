import logging

from app_logging import _sanitize_log_text, LogSanitizingFilter, configure_logging

_CHECK_MARK = chr(0x2705)
_MUSIC_NOTE = chr(0x1F3B5)
_ACCENTED = "caf" + chr(0xE9) + " se" + chr(0xF1) + "or " + chr(0xFC) + "ber"


class TestSanitizeLogText:
    def test_removes_emoji_and_symbols(self):
        assert _sanitize_log_text("done " + _CHECK_MARK) == "done"
        assert _sanitize_log_text("track " + _MUSIC_NOTE + " ready") == "track ready"

    def test_newline_becomes_space(self):
        assert _sanitize_log_text("hello\nworld") == "hello world"

    def test_crlf_collapses_to_single_space(self):
        assert _sanitize_log_text("hello\r\nworld") == "hello world"

    def test_control_chars_become_space(self):
        assert _sanitize_log_text("a\x00b\x07c\x7f") == "a b c"

    def test_unicode_line_separators_become_space(self):
        for sep in (chr(0x85), chr(0x2028), chr(0x2029)):
            assert _sanitize_log_text("a" + sep + "b") == "a b"

    def test_unicode_separator_cannot_forge_a_line(self):
        forged = "user42" + chr(0x2028) + "[INFO]-[fake]-dropped all tables"
        result = _sanitize_log_text(forged)
        assert all(sep not in result for sep in (chr(0x85), chr(0x2028), chr(0x2029)))
        assert len(result.splitlines()) == 1

    def test_tab_is_preserved(self):
        assert _sanitize_log_text("col1\tcol2") == "col1\tcol2"

    def test_latin1_accents_pass_through(self):
        assert _sanitize_log_text(_ACCENTED) == _ACCENTED

    def test_log_injection_cannot_forge_a_line(self):
        forged = "user42\n[INFO]-[fake]-dropped all tables"
        result = _sanitize_log_text(forged)
        assert "\n" not in result
        assert "\r" not in result
        assert result == "user42 [INFO]-[fake]-dropped all tables"

    def test_non_string_returned_unchanged(self):
        assert _sanitize_log_text(123) == 123
        assert _sanitize_log_text(None) is None


class TestLogSanitizingFilter:
    def _record(self, msg, args=None):
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg=msg, args=args, exc_info=None,
        )

    def test_sanitizes_msg(self):
        record = self._record("oops\ninjected " + _CHECK_MARK)
        LogSanitizingFilter().filter(record)
        assert record.msg == "oops injected"

    def test_sanitizes_tuple_args_leaving_non_strings(self):
        record = self._record("%s %s", args=("a\nb", 7))
        LogSanitizingFilter().filter(record)
        assert record.args == ("a b", 7)

    def test_sanitizes_dict_args(self):
        record = self._record("%(x)s")
        record.args = {"x": "p\nq", "n": 3}
        LogSanitizingFilter().filter(record)
        assert record.args == {"x": "p q", "n": 3}

    def test_filter_always_returns_true(self):
        assert LogSanitizingFilter().filter(self._record("hi")) is True


class TestConfigureLogging:
    def test_attaches_sanitizing_filter_once(self):
        root = logging.getLogger()
        saved = {handler: list(handler.filters) for handler in root.handlers}
        try:
            configure_logging()
            configure_logging()
            assert root.handlers
            for handler in root.handlers:
                count = sum(isinstance(f, LogSanitizingFilter) for f in handler.filters)
                assert count == 1
        finally:
            for handler in root.handlers:
                if handler in saved:
                    handler.filters = saved[handler]
                else:
                    handler.filters = [
                        f for f in handler.filters
                        if not isinstance(f, LogSanitizingFilter)
                    ]
