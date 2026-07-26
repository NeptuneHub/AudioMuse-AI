# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Issue #784: a worker deregistered by a transient Redis outage rejoins on heartbeat.

When Redis is unreachable longer than the worker-key TTL, the key expires,
clean_worker_registry SREMs the worker from rq:workers, and no rq release through
2.10 ever re-adds it - historically the live worker kept taking jobs while invisible
to the dashboard. These tests drive the real worker classes against an in-memory
Redis stand-in and assert the heartbeat re-registers. They are written to pass on
both rq pins the project ships: 2.10 (docker, native builds) and 2.7 (noavx2 image),
whose heartbeats queue the same commands in different orders.

Main Features:
* Both the forking Worker and the Windows SimpleWorker carry the mixin and rejoin.
* A heartbeat re-adds the worker to rq:workers and rq:workers:<queue>, clears a
  false death stamp, re-EXPIREs the key so it always regains a TTL, and restores
  identity fields lost to an idle-time outage wherever rq offers serialize().
* Nothing is ever queued into rq's own heartbeat pipeline: rq reads that pipeline's
  results positionally, and an injected command would make it inspect the wrong
  result and delete the running job's key.
* The beat thread is serialized with execution preparation and cleanup, refreshes
  the worker key when no execution exists, and execute_job joins it before returning.
"""

import logging
import threading
import types

import pytest
from rq.utils import now

import rq_heartbeat_worker as rhw


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sets = {}
        self.ttls = {}
        self.connection_pool = types.SimpleNamespace(connection_kwargs={})

    def hset(self, name, key=None, value=None, mapping=None):
        h = self.hashes.setdefault(name, {})
        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in h:
                    added += 1
                h[k] = '' if v is None else str(v)
        if key is not None:
            if key not in h:
                added += 1
            h[key] = '' if value is None else str(value)
        return added

    def hexists(self, name, key):
        return key in self.hashes.get(name, {})

    def hdel(self, name, *keys):
        h = self.hashes.get(name, {})
        removed = 0
        for k in keys:
            if k in h:
                del h[k]
                removed += 1
        return removed

    def sadd(self, name, *members):
        s = self.sets.setdefault(name, set())
        before = len(s)
        s.update(members)
        return len(s) - before

    def srem(self, name, *members):
        s = self.sets.setdefault(name, set())
        before = len(s)
        for m in members:
            s.discard(m)
        return before - len(s)

    def smembers(self, name):
        return set(self.sets.get(name, set()))

    def scard(self, name):
        return len(self.sets.get(name, set()))

    def exists(self, *keys):
        return sum(1 for k in keys if k in self.hashes or k in self.sets)

    def expire(self, name, ttl):
        if name not in self.hashes and name not in self.sets:
            return 0
        self.ttls[name] = ttl
        return 1

    def delete(self, *keys):
        removed = 0
        for k in keys:
            if k in self.hashes:
                del self.hashes[k]
                removed += 1
            if k in self.sets:
                del self.sets[k]
                removed += 1
            self.ttls.pop(k, None)
        return removed

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, parent):
        self.parent = parent
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def hset(self, *args, **kwargs):
        self.commands.append(('hset', args, kwargs))

    def hdel(self, *args, **kwargs):
        self.commands.append(('hdel', args, kwargs))

    def sadd(self, *args, **kwargs):
        self.commands.append(('sadd', args, kwargs))

    def expire(self, *args, **kwargs):
        self.commands.append(('expire', args, kwargs))

    def execute(self):
        results = [
            getattr(self.parent, name)(*args, **kwargs) for name, args, kwargs in self.commands
        ]
        self.commands = []
        return results


def _make_worker(worker_cls):
    fake = FakeRedis()
    worker = worker_cls(
        ['default'],
        connection=fake,
        prepare_for_work=False,
        worker_ttl=120,
        job_monitoring_interval=30,
        name='w784',
    )
    worker.birth_date = now()
    return worker, fake


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_partial_key_after_outage_rejoins_registry_on_heartbeat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale', 'successful_job_count': '400'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_reregistration_is_idempotent_when_already_registered(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'birth': 'ORIGINAL', 'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = {worker.key}

    worker.heartbeat()

    assert fake.scard('rq:workers') == 1
    assert fake.hashes[worker.key]['birth'] == 'ORIGINAL'


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_never_queues_repair_commands_into_rqs_pipeline(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()
    pipe = FakePipeline(FakeRedis())

    worker.heartbeat(pipeline=pipe)

    assert len(pipe.commands) == 2, (
        "rq's maintain_heartbeats reads the pipeline it hands to heartbeat "
        "positionally (results[7] on 2.7, the results[0] recreation check on 2.10); "
        "an extra command shifts those positions and rq can delete a RUNNING job's key"
    )
    assert not [name for name, _args, _kwargs in pipe.commands if name in ('sadd', 'hdel')]
    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_unpipelined_heartbeat_recreates_expired_key_with_ttl_and_registration(worker_cls):
    worker, fake = _make_worker(worker_cls)

    worker.heartbeat()

    assert fake.hashes[worker.key].get('last_heartbeat')
    assert fake.ttls.get(worker.key) == worker.worker_ttl + 60, (
        "rq 2.7's own heartbeat EXPIREs before it HSETs, so an expired key would be "
        "recreated without a TTL and live forever; the repair EXPIRE must land after"
    )
    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_repair_registration_death_clear_and_expire_run_through_one_pipeline(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()
    seen = []
    original_pipeline = fake.pipeline

    def tracking_pipeline():
        pipe = original_pipeline()
        original_execute = pipe.execute

        def tracking_execute():
            seen.append([name for name, args, kwargs in pipe.commands])
            return original_execute()

        pipe.execute = tracking_execute
        return pipe

    fake.pipeline = tracking_pipeline

    worker.heartbeat()

    if hasattr(worker, 'serialize'):
        assert ['hset', 'sadd', 'sadd', 'hdel', 'expire'] in seen
    else:
        assert ['sadd', 'sadd', 'hdel', 'expire'] in seen
    assert fake.ttls.get(worker.key)


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_failed_reregistration_is_logged_and_leaves_the_heartbeat_intact(
    worker_cls, monkeypatch, caplog
):
    worker, fake = _make_worker(worker_cls)

    def fail_registration(*args, **kwargs):
        raise RuntimeError('registration failed')

    monkeypatch.setattr(rhw.worker_registration, 'register', fail_registration)

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker.heartbeat()

    assert fake.hashes[worker.key].get('last_heartbeat')
    assert worker.key not in fake.smembers('rq:workers')
    assert any('re-registration on heartbeat failed' in r.message for r in caplog.records)


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_reraises_stop_requested_for_warm_shutdown(worker_cls, monkeypatch):
    worker, _fake = _make_worker(worker_cls)

    def raise_stop(*args, **kwargs):
        raise rhw.StopRequested()

    monkeypatch.setattr(rhw.worker_registration, 'register', raise_stop)

    with pytest.raises(rhw.StopRequested):
        worker.heartbeat()


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_idle_reconnect_restores_identity_fields_from_rqs_own_serializer(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    if hasattr(worker, 'serialize'):
        assert fake.hashes[worker.key].get('birth'), (
            "rq rebuilds a recreated key only from maintain_heartbeats (mid-job), so "
            "an outage that expires the key while the worker is idle leaves blank "
            "identity fields forever; the repair must reuse rq's own serialize()"
        )
        assert fake.hashes[worker.key].get('queues') == 'default'
    else:
        assert 'birth' not in fake.hashes[worker.key], (
            "rq 2.7 (noavx2) has no Worker.serialize(); the accepted degradation is a "
            "partial hash until restart, never a hand-rolled identity mapping (#799)"
        )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_identity_rebuild_is_skipped_before_birth_registration(worker_cls, caplog):
    worker, fake = _make_worker(worker_cls)
    worker.birth_date = None
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker.heartbeat()

    assert 'birth' not in fake.hashes[worker.key], (
        "serialize() asserts birth_date is set, so a heartbeat that fires before "
        "register_birth must skip the rebuild instead of failing the whole repair"
    )
    assert not caplog.records
    assert worker.key in fake.smembers('rq:workers')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_repair_expire_honours_the_ttl_rq_asked_for(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat(timeout=90)

    assert fake.ttls[worker.key] == 90, (
        "the mixin must not override the caller's timeout with worker_ttl + 60"
    )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_live_worker_falsely_marked_dead_by_the_janitor_rejoins_on_the_next_beat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'death': 'STAMPED_BY_JANITOR'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    assert worker.key in fake.smembers('rq:workers'), (
        "rq_janitor calls register_death on any worker whose last_heartbeat reads "
        "None, which can falsely hit a live worker whose hash was recreated by a "
        "job-count increment; skipping re-registration on death would be permanent "
        "because nothing clears death outside register_birth at process start"
    )
    assert 'death' not in fake.hashes[worker.key], (
        "only a live worker can reach this code, since a genuinely dead one has "
        "exited and stopped beating, so a death stamp here is false and must be "
        "cleared rather than left to contradict the heartbeat we just wrote"
    )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_survives_when_last_heartbeat_attribute_was_never_set(worker_cls, caplog):
    worker, fake = _make_worker(worker_cls)
    if hasattr(worker, 'last_heartbeat'):
        del worker.last_heartbeat
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker.heartbeat()

    assert worker.key in fake.smembers('rq:workers'), (
        "issue #799: rq 2.7 assigns last_heartbeat only in refresh(), so any repair "
        "code reading the attribute directly crashes a worker that never refreshed"
    )
    assert not caplog.records


def test_heartbeat_waits_until_execution_preparation_commits(monkeypatch):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    job = types.SimpleNamespace(id='job-preparing')
    stop = threading.Event()
    heartbeat_lock = threading.Lock()
    worker._execution_heartbeat_stop = stop
    worker._execution_heartbeat_lock = heartbeat_lock
    preparation_entered = threading.Event()
    release_preparation = threading.Event()
    heartbeat_done = threading.Event()
    order = []

    def base_prepare(self, _job):
        self.execution = object()
        order.append('prepare-start')
        preparation_entered.set()
        release_preparation.wait(2)
        order.append('prepare-end')
        return self.execution

    def maintain(_job):
        order.append('heartbeat')
        heartbeat_done.set()

    monkeypatch.setattr(rhw.SimpleWorker, 'prepare_execution', base_prepare)
    worker.maintain_heartbeats = maintain

    preparer = threading.Thread(target=worker.prepare_execution, args=(job,))
    preparer.start()
    assert preparation_entered.wait(1)

    beater = threading.Thread(
        target=worker._refresh_job_heartbeat,
        args=(job, stop, heartbeat_lock),
    )
    beater.start()
    assert not heartbeat_done.wait(0.05), (
        "a heartbeat must not use an execution before its creation transaction commits"
    )

    release_preparation.set()
    preparer.join(1)
    beater.join(1)

    assert not preparer.is_alive()
    assert not beater.is_alive()
    assert order == ['prepare-start', 'prepare-end', 'heartbeat']


def test_cleanup_waits_for_an_inflight_heartbeat(monkeypatch):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    old_execution = object()
    worker.execution = old_execution
    job = types.SimpleNamespace(id='job-teardown')
    stop = threading.Event()
    heartbeat_lock = threading.Lock()
    worker._execution_heartbeat_stop = stop
    worker._execution_heartbeat_lock = heartbeat_lock
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()
    cleanup_done = threading.Event()
    order = []

    def maintain(_job):
        assert worker.execution is old_execution
        order.append('heartbeat-start')
        heartbeat_entered.set()
        release_heartbeat.wait(2)
        order.append('heartbeat-end')

    def base_cleanup(self, _job, _pipeline):
        order.append('cleanup')
        self.execution = None

    worker.maintain_heartbeats = maintain
    monkeypatch.setattr(rhw.SimpleWorker, 'cleanup_execution', base_cleanup)

    beater = threading.Thread(
        target=worker._refresh_job_heartbeat,
        args=(job, stop, heartbeat_lock),
    )
    beater.start()
    assert heartbeat_entered.wait(1)

    def cleanup():
        try:
            worker.cleanup_execution(job, object())
        finally:
            cleanup_done.set()

    cleaner = threading.Thread(target=cleanup)
    cleaner.start()
    assert stop.wait(1), "cleanup must stop future beats before waiting on the lock"
    assert not cleanup_done.wait(0.05), (
        "cleanup must not delete the execution while its heartbeat is in flight"
    )

    release_heartbeat.set()
    beater.join(1)
    cleaner.join(1)

    assert not beater.is_alive()
    assert not cleaner.is_alive()
    assert order == ['heartbeat-start', 'heartbeat-end', 'cleanup']
    assert worker.execution is None


def test_beat_thread_still_logs_a_genuine_heartbeat_failure(caplog):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = object()
    job = types.SimpleNamespace(id='job-real-failure')
    stop = threading.Event()
    heartbeat_lock = threading.Lock()

    def fail(_job):
        raise OSError('redis is down')

    worker.maintain_heartbeats = fail
    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker._refresh_job_heartbeat(job, stop, heartbeat_lock)

    assert any('Heartbeat refresh failed' in r.message for r in caplog.records), (
        "serialization with cleanup must not silence a real Redis outage"
    )


def test_beat_skips_execution_work_before_the_execution_is_prepared():
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = None
    job = types.SimpleNamespace(id='job-not-prepared')
    called = []
    worker.maintain_heartbeats = lambda _job: called.append(True)

    worker._refresh_job_heartbeat(job, threading.Event(), threading.Lock())

    assert not called


def test_beat_still_refreshes_the_worker_key_while_no_execution_exists():
    worker, fake = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = None
    fake.hashes[worker.key] = {'birth': 'ORIGINAL', 'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = {worker.key}
    job = types.SimpleNamespace(id='job-in-teardown')

    worker._refresh_job_heartbeat(job, threading.Event(), threading.Lock())

    assert fake.hashes[worker.key]['last_heartbeat'] != 'stale', (
        "cleanup_execution nulls the execution partway through handle_job_success, so a "
        "beat that returns early there stops refreshing the worker key too, and the key "
        "expires on the 90s TTL maintain_heartbeats left it with"
    )
    assert fake.ttls[worker.key] == worker.job_monitoring_interval + 60


def test_beat_thread_logs_instead_of_dying_when_a_heartbeat_raises_stop_requested(caplog):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = object()
    job = types.SimpleNamespace(id='job-stop-requested')

    def raise_stop(_job):
        raise rhw.StopRequested()

    worker.maintain_heartbeats = raise_stop
    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker._refresh_job_heartbeat(job, threading.Event(), threading.Lock())

    assert any('Heartbeat refresh failed' in r.message for r in caplog.records), (
        "rq raises StopRequested from its signal handler, which Python only ever runs on "
        "the main thread, so re-raising it here cannot deliver a shutdown request - it "
        "would only kill the beat thread through threading.excepthook with no app log"
    )


def test_execute_job_waits_until_its_heartbeat_thread_has_exited(monkeypatch):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    job = types.SimpleNamespace(id='job-blocked-beat')
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()
    heartbeat_exited = threading.Event()
    execute_done = threading.Event()
    result = []

    def blocked_loop(self, _job, _stop, _heartbeat_lock):
        heartbeat_entered.set()
        release_heartbeat.wait(2)
        heartbeat_exited.set()

    monkeypatch.setattr(rhw.HeartbeatSimpleWorker, '_heartbeat_loop', blocked_loop)
    monkeypatch.setattr(rhw.SimpleWorker, 'execute_job', lambda self, _job, _queue: 'finished')

    def execute():
        try:
            result.append(worker.execute_job(job, None))
        finally:
            execute_done.set()

    execution_thread = threading.Thread(target=execute)
    execution_thread.start()
    assert heartbeat_entered.wait(1)
    assert worker._execution_heartbeat_stop.wait(1)
    assert not execute_done.wait(0.05), (
        "execute_job must not return while an old heartbeat can still reach the next job"
    )

    release_heartbeat.set()
    execution_thread.join(1)

    assert not execution_thread.is_alive()
    assert heartbeat_exited.is_set()
    assert result == ['finished']
    assert worker._execution_heartbeat_stop is None
    assert worker._execution_heartbeat_lock is None
