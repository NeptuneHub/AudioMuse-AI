# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cross-process election for the RQ janitor cycle.

Two janitors probing the same abandoned job both decided to requeue it, because
retrying had no claim anywhere. The cycle lock elects one owner per pass; an
unelected janitor must skip the whole cycle rather than run a partial one, and a
database that cannot be reached must fail CLOSED so nobody runs.

Main Features:
* The cycle lock fails closed when the database is unavailable
* An unelected janitor never touches the started-job registry
"""

from contextlib import contextmanager
from unittest.mock import MagicMock


def test_janitor_cycle_lock_fails_closed_when_database_is_unavailable(monkeypatch):
    from tasks import multiserver_sync as sync

    monkeypatch.setattr(
        sync, 'connect_raw', lambda: (_ for _ in ()).throw(RuntimeError('db down'))
    )

    with sync.janitor_cycle_lock() as owns_cycle:
        assert owns_cycle is False


def test_unelected_janitor_never_runs_started_registry_cleanup(monkeypatch):
    import rq_janitor

    @contextmanager
    def not_elected():
        yield False

    queue = MagicMock()
    monkeypatch.setattr(rq_janitor, 'janitor_cycle_lock', not_elected)

    assert rq_janitor.run_elected_janitor_cycle([queue]) is False
    queue.started_job_registry.cleanup.assert_not_called()


def test_elected_janitor_runs_started_registry_cleanup_once(monkeypatch):
    import rq_janitor

    @contextmanager
    def elected():
        yield True

    queue = MagicMock()
    queue.started_job_registry.count = 0
    queue.finished_job_registry.count = 0
    queue.failed_job_registry.count = 0
    monkeypatch.setattr(rq_janitor, 'janitor_cycle_lock', elected)
    monkeypatch.setattr(rq_janitor, 'recover_abandoned_sweeps', lambda: None)
    monkeypatch.setattr(
        rq_janitor, 'recover_provider_migration_restart_handshakes', lambda: None
    )
    monkeypatch.setattr(rq_janitor, 'reap_orphaned_tasks', lambda: None)
    monkeypatch.setattr(rq_janitor, 'clean_worker_registry', lambda queue: None)
    monkeypatch.setattr(rq_janitor.Worker, 'all', lambda connection=None: [])
    redis = MagicMock()
    redis.scard.return_value = 0
    redis.scan.return_value = (0, [])
    monkeypatch.setattr(rq_janitor, 'redis_conn', redis)

    assert rq_janitor.run_elected_janitor_cycle([queue]) is True
    queue.started_job_registry.cleanup.assert_called_once()
