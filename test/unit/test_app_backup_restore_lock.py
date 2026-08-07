# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_backup restore lock file.

Drives the real lock helpers against a temporary backup directory to prove the
lock fails closed: a lock that cannot be read counts as held and refuses a new
restore, while a readable one still expires on the TTL.

Main Features:
* An unreadable lock blocks a new restore and is left untouched on disk.
* An unreadable lock reads as held, so a chunked upload is not overtaken.
* A readable lock older than the TTL is still cleared and re-taken.
* A missing lock is still acquired and a fresh one still refuses a second
  restore.
"""

import builtins
import logging
import time

import pytest

import app_backup


@pytest.fixture
def backup_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
    return tmp_path


def _lock_file(backup_dir):
    return backup_dir / '.restore.lock'


def _write_lock(backup_dir, age_seconds):
    _lock_file(backup_dir).write_text(str(time.time() - age_seconds), encoding='utf-8')


def _make_lock_unreadable(patcher, backup_dir):
    lock_path = str(_lock_file(backup_dir))
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if str(file) == lock_path:
            raise PermissionError(13, 'Permission denied', lock_path)
        return real_open(file, *args, **kwargs)

    patcher.setattr(builtins, 'open', fake_open)


class TestRestoreLockFailsClosed:
    def test_unreadable_lock_blocks_a_new_restore_and_never_expires_on_the_ttl(
        self, backup_dir, monkeypatch
    ):
        _write_lock(backup_dir, age_seconds=app_backup.RESTORE_LOCK_TTL_SECONDS * 10)
        held_by = _lock_file(backup_dir).read_text(encoding='utf-8')

        with monkeypatch.context() as unreadable:
            _make_lock_unreadable(unreadable, backup_dir)
            assert app_backup._acquire_restore_lock() is False
            assert app_backup._restore_lock_age() == app_backup.RESTORE_LOCK_UNREADABLE_AGE
            assert app_backup.RESTORE_LOCK_UNREADABLE_AGE <= app_backup.RESTORE_LOCK_TTL_SECONDS

        assert _lock_file(backup_dir).read_text(encoding='utf-8') == held_by

    def test_unreadable_lock_counts_as_held(self, backup_dir, monkeypatch):
        _write_lock(backup_dir, age_seconds=5)

        with monkeypatch.context() as unreadable:
            _make_lock_unreadable(unreadable, backup_dir)
            assert app_backup._restore_lock_held() is True

    def test_unreadable_lock_is_logged_with_its_traceback(
        self, backup_dir, monkeypatch, caplog
    ):
        _write_lock(backup_dir, age_seconds=5)

        with monkeypatch.context() as unreadable:
            _make_lock_unreadable(unreadable, backup_dir)
            with caplog.at_level(logging.ERROR, logger='app_backup'):
                age = app_backup._restore_lock_age()

        assert age == app_backup.RESTORE_LOCK_UNREADABLE_AGE
        assert any(record.exc_info for record in caplog.records)


class TestRestoreLockStillExpires:
    def test_readable_lock_older_than_the_ttl_still_expires(self, backup_dir):
        _write_lock(backup_dir, age_seconds=app_backup.RESTORE_LOCK_TTL_SECONDS + 60)
        stale = _lock_file(backup_dir).read_text(encoding='utf-8')

        assert app_backup._restore_lock_held() is False
        assert app_backup._acquire_restore_lock() is True
        assert _lock_file(backup_dir).read_text(encoding='utf-8') != stale
        assert app_backup._restore_lock_held() is True

    def test_fresh_readable_lock_refuses_a_second_restore(self, backup_dir):
        _write_lock(backup_dir, age_seconds=5)

        assert app_backup._acquire_restore_lock() is False
        assert app_backup._restore_lock_held() is True

    def test_missing_lock_is_acquired_and_is_not_held_before_that(self, backup_dir):
        assert app_backup._restore_lock_held() is False
        assert app_backup._acquire_restore_lock() is True
        assert _lock_file(backup_dir).exists()

    def test_released_lock_lets_the_next_restore_start(self, backup_dir):
        assert app_backup._acquire_restore_lock() is True
        app_backup._release_restore_lock()

        assert _lock_file(backup_dir).exists() is False
        assert app_backup._restore_lock_held() is False
        assert app_backup._acquire_restore_lock() is True
