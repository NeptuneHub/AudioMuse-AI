# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The embedded PostgreSQL stops even after an earlier supervisor died hard.

pgserver stops the server only when its on-disk handle list holds nothing but
the current pid. A supervisor that was killed never removes its pid, so every
later clean stop silently skipped the shutdown and left postgres running on
Windows, Linux and macOS alike.

Main Features:
* Dead pids, recycled pids (a role child, a foreign program) and garbage
  entries are dropped before pgserver's cleanup decides, under its own lock.
* A broken handle file never prevents the cleanup call itself.
* Starting the server prunes the list too, so the file stays honest.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest


class _Handles:
    def __init__(self, pids):
        self.values = list(pids)

    def get(self):
        return list(self.values)

    def put(self, values):
        self.values = list(values)


class _Lock:
    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_exc):
        return False


def _fake_psutil(alive):
    mod = types.ModuleType('psutil')
    mod.NoSuchProcess = type('NoSuchProcess', (Exception,), {})
    mod.AccessDenied = type('AccessDenied', (Exception,), {})

    class Process:
        def __init__(self, pid):
            if pid not in alive:
                raise mod.NoSuchProcess(pid)
            self._exe, self._cmdline = alive[pid]

        def exe(self):
            return self._exe

        def cmdline(self):
            return self._cmdline

    mod.Process = Process
    return mod


@pytest.fixture
def database(monkeypatch):
    import database

    monkeypatch.setattr(database, '_embedded_server', None)
    return database


class TestStoppingTheEmbeddedPostgresAfterAnEarlierCrash:
    def test_only_live_supervisors_of_this_executable_keep_their_handle(self, monkeypatch, database):
        exe = sys.executable
        monkeypatch.setattr(database, 'psutil', _fake_psutil({
            1000: (exe, [exe, 'start']),
            3000: ('/usr/lib/systemd/systemd', ['/usr/lib/systemd/systemd']),
            4000: (exe, [exe, '--role=worker-default']),
        }))
        server = MagicMock()
        server._lock = _Lock()
        server.global_process_id_list = _Handles([2000, 1000, 3000, 4000, None, 'x', os.getpid()])
        seen = []
        server.cleanup.side_effect = lambda: seen.append(server.global_process_id_list.get())
        monkeypatch.setattr(database, '_embedded_server', server)

        database.stop_embedded()

        assert seen == [[1000, os.getpid()]], (
            'a supervisor killed hard left its pid in the handle list, so pgserver '
            'saw another holder and skipped pg_ctl stop: postgres outlived every '
            'later clean stop; a pid recycled by a role child or a foreign program '
            'and a garbage entry are not holders either'
        )
        assert server._lock.entered == 1, (
            'pgserver mutates the same file under its inter-process lock; an '
            'unlocked read-modify-write could erase a holder another instance just added'
        )
        assert database._embedded_server is None

    def test_a_broken_handle_file_never_blocks_the_cleanup(self, monkeypatch, database):
        server = MagicMock()
        server.global_process_id_list.get.side_effect = ValueError('garbage in the handle file')
        monkeypatch.setattr(database, '_embedded_server', server)

        database.stop_embedded()

        server.cleanup.assert_called_once_with()

    def test_starting_the_server_prunes_the_list_too(self, monkeypatch, database):
        monkeypatch.setattr(database, 'psutil', _fake_psutil({}))
        server = MagicMock()
        server.global_process_id_list = _Handles([4000, os.getpid()])
        server.get_uri.return_value = 'postgresql://x'
        fake = types.ModuleType('pgserver')
        fake.get_server = lambda data_dir: server
        monkeypatch.setitem(sys.modules, 'pgserver', fake)

        assert database.start_embedded('/data/pgdata') == 'postgresql://x'

        assert server.global_process_id_list.get() == [os.getpid()]
        assert database._embedded_server is server
