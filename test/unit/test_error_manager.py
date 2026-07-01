# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Error registry, classification, and safe message building in error_manager.

Covers the code registry, build/from_exception/classify/record, and the guarantee
that no traceback or raw exception text ever reaches the returned message.

Main Features:
* Registry completeness and unknown-code fallbacks for class and message
* Detail appended as a single truncated line; unknown codes suppress caller detail
* classify maps exception names via MRO; record logs the full trace to the logger
  only, and http_status_for_code maps error classes to HTTP statuses
"""

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from error import error_dictionary as ed
from error import error_manager as em


def _named_codes():
    return [
        value
        for name, value in vars(ed).items()
        if (name.startswith('ERR_') or name == 'UNKNOWN_ERROR_CODE') and isinstance(value, int)
    ]


class TestRegistry:
    def test_every_entry_has_class_and_message(self):
        for code, entry in ed.ERROR_REGISTRY.items():
            assert isinstance(code, int)
            assert entry['error_class'].strip()
            assert entry['default_message'].strip()

    def test_every_named_constant_is_registered(self):
        for code in _named_codes():
            assert code in ed.ERROR_REGISTRY

    def test_unknown_code_helpers_fall_back(self):
        assert ed.get_error_class(123456) == 'Unknown Error'
        assert (
            ed.get_default_message(123456)
            == ed.ERROR_REGISTRY[ed.UNKNOWN_ERROR_CODE]['default_message']
        )


class TestBuild:
    def test_default_message(self):
        result = em.build(ed.ERR_ANALYSIS_FAILED)
        assert result['error_code'] == ed.ERR_ANALYSIS_FAILED
        assert result['error_class'] == 'Analysis Error'
        assert result['error_message'] == ed.get_default_message(ed.ERR_ANALYSIS_FAILED)

    def test_appended_detail_is_single_line(self):
        result = em.build(ed.ERR_ANALYSIS_FAILED, 'first line\nsecond\tline')
        assert '\n' not in result['error_message']
        assert '\t' not in result['error_message']
        assert 'first line' in result['error_message']
        assert result['error_message'].startswith(ed.get_default_message(ed.ERR_ANALYSIS_FAILED))

    def test_long_detail_is_truncated(self):
        result = em.build(ed.ERR_ANALYSIS_FAILED, 'x' * 2000)
        assert len(result['error_message']) < 600
        assert result['error_message'].endswith('...')

    def test_unknown_code_falls_back(self):
        result = em.build(987654)
        assert result['error_code'] == ed.UNKNOWN_ERROR_CODE
        assert result['error_class'] == 'Unknown Error'

    def test_unknown_message_points_to_container_logs(self):
        result = em.build(987654)
        assert 'log' in result['error_message'].lower()

    def test_unknown_code_suppresses_caller_detail(self):
        result = em.build(ed.UNKNOWN_ERROR_CODE, 'leak this secret detail')
        assert 'leak this secret detail' not in result['error_message']
        assert result['error_message'] == ed.get_default_message(ed.UNKNOWN_ERROR_CODE)


class TestClassify:
    def test_known_exception_name_maps_to_code(self):
        class ReadTimeout(Exception):
            pass

        assert (
            em.classify(ReadTimeout('slow'), ed.ERR_ANALYSIS_FAILED) == ed.ERR_MEDIASERVER_TIMEOUT
        )

    def test_unknown_exception_uses_default(self):
        assert em.classify(ValueError('x'), ed.ERR_ANALYSIS_FAILED) == ed.ERR_ANALYSIS_FAILED

    def test_subclass_matches_parent_via_mro(self):
        class OperationalError(Exception):
            pass

        class MyCustomDbError(OperationalError):
            pass

        assert em.classify(MyCustomDbError('x'), ed.ERR_ANALYSIS_FAILED) == ed.ERR_DB_CONNECTION

    def test_audiomuse_error_keeps_its_code(self):
        exc = em.AudioMuseError(ed.ERR_DB_CONNECTION, 'down')
        assert em.classify(exc, ed.ERR_ANALYSIS_FAILED) == ed.ERR_DB_CONNECTION


class TestFromException:
    def test_audiomuse_error_round_trips(self):
        exc = em.AudioMuseError(ed.ERR_DB_CONNECTION, 'connection refused')
        result = em.from_exception(exc)
        assert result['error_code'] == ed.ERR_DB_CONNECTION
        assert result['error_class'] == 'Database Error'
        assert 'connection refused' in result['error_message']

    def test_generic_exception_defaults_to_unknown_and_hides_detail(self):
        result = em.from_exception(ValueError('boom\nmore'))
        assert result['error_code'] == ed.UNKNOWN_ERROR_CODE
        assert result['error_class'] == 'Unknown Error'
        assert '\n' not in result['error_message']
        assert 'boom' not in result['error_message']
        assert result['error_message'] == ed.get_default_message(ed.UNKNOWN_ERROR_CODE)

    def test_explicit_code_is_used(self):
        result = em.from_exception(ValueError('boom'), code=ed.ERR_ANALYSIS_FAILED)
        assert result['error_code'] == ed.ERR_ANALYSIS_FAILED


class TestNoStackInMessage:
    def test_message_never_contains_newline(self):
        try:
            raise RuntimeError('line one\nline two\nline three')
        except RuntimeError as exc:
            built = em.build(ed.ERR_ANALYSIS_FAILED, str(exc))
            from_exc = em.from_exception(exc, code=ed.ERR_ANALYSIS_FAILED)
        assert '\n' not in built['error_message']
        assert '\n' not in from_exc['error_message']


class TestNoTracebackEverLeaks:
    def test_record_never_returns_traceback(self):
        try:
            raise ValueError('boom')
        except ValueError as exc:
            result = em.record(ed.ERR_ANALYSIS_FAILED, str(exc), exc=exc)
        assert 'traceback' not in result
        assert '\n' not in result['error_message']

    def test_from_exception_never_returns_traceback(self):
        try:
            raise ValueError('boom')
        except ValueError as exc:
            result = em.from_exception(exc)
        assert 'traceback' not in result

    def test_record_logs_full_trace_to_given_logger(self):
        import logging as _logging

        class _Capture(_logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(record)

        cap = _Capture()
        test_logger = _logging.getLogger('test_error_manager_capture')
        test_logger.addHandler(cap)
        test_logger.setLevel(_logging.ERROR)
        try:
            raise ValueError('boom')
        except ValueError as exc:
            em.record(ed.ERR_ANALYSIS_FAILED, str(exc), exc=exc, logger=test_logger)
        test_logger.removeHandler(cap)
        assert cap.records and cap.records[0].exc_info is not None

    def test_audiomuse_to_dict_is_exactly_three_keys(self):
        exc = em.AudioMuseError(ed.ERR_DB_CONNECTION, 'down')
        assert set(exc.to_dict().keys()) == {'error_code', 'error_class', 'error_message'}


class TestAudioMuseError:
    def test_str_returns_message(self):
        exc = em.AudioMuseError(ed.ERR_MEDIASERVER_UNREACHABLE, 'host down')
        assert str(exc) == exc.error_message
        assert 'host down' in str(exc)

    def test_unknown_code_normalized(self):
        exc = em.AudioMuseError(424242)
        assert exc.code == ed.UNKNOWN_ERROR_CODE
        assert exc.to_dict()['error_class'] == 'Unknown Error'


class TestHttpStatus:
    def test_media_server_is_bad_gateway(self):
        assert em.http_status_for_code(ed.ERR_MEDIASERVER_UNREACHABLE) == 502

    def test_config_is_bad_request(self):
        assert em.http_status_for_code(ed.ERR_CONFIG_INVALID) == 400

    def test_database_is_service_unavailable(self):
        assert em.http_status_for_code(ed.ERR_DB_CONNECTION) == 503

    def test_other_is_internal_error(self):
        assert em.http_status_for_code(ed.ERR_ANALYSIS_FAILED) == 500
