# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Issue #784: a worker deregistered by a transient Redis outage rejoins on heartbeat.

When Redis is unreachable longer than the worker-key TTL, the key expires,
clean_worker_registry SREMs the worker from rq:workers, and on reconnect the
heartbeat recreates only a partial hash - historically the live worker kept taking
jobs while invisible to the dashboard. These tests drive the real worker classes
against an in-memory Redis stand-in and assert the heartbeat re-registers.

Main Features:
* Both the forking Worker and the Windows SimpleWorker carry the mixin and rejoin.
* A heartbeat re-adds the worker to rq:workers and rq:workers:<queue> and restores
  the identity hash (birth/hostname/queues) when the key came back partial.
* Re-registration is idempotent and does not rewrite the hash during normal beats.
* Nothing is ever queued into rq's own heartbeat pipeline: rq 2.7.0 reads that
  pipeline's results by fixed index, and an injected command would make it delete
  the running job's key.
* The identity rebuild works when worker.last_heartbeat was never set: rq 2.7.0
  assigns that attribute only in refresh() (issue #799).
* The repair HSET and EXPIRE run through one pipeline on the raw connection and
  the recreated key always carries a TTL.
"""

import logging
import threading
import types

import pytest
from rq.utils import now, utcformat

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
    worker.hostname = 'testhost'
    worker.pid = 4321
    worker.ip_address = '10.0.0.9'
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
def test_partial_key_identity_hash_is_restored_on_heartbeat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale', 'successful_job_count': '400'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    restored = fake.hashes[worker.key]
    assert restored.get('birth')
    assert restored.get('hostname') == 'testhost'
    assert restored.get('queues') == 'default'


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_reregistration_is_idempotent_when_already_registered(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'birth': 'ORIGINAL', 'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = {worker.key}

    worker.heartbeat()

    assert fake.scard('rq:workers') == 1
    assert fake.hashes[worker.key]['birth'] == 'ORIGINAL'


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_pipelined_heartbeat_never_queues_into_rqs_pipeline(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()
    pipe = FakePipeline(FakeRedis())

    worker.heartbeat(pipeline=pipe)

    assert [name for name, _args, _kwargs in pipe.commands] == ['hset', 'expire'], (
        "rq 2.7.0 reads maintain_heartbeats pipeline results by FIXED index "
        "(results[7] = job.heartbeat); the override must retain exactly two commands "
        "or RQ can inspect the wrong result and delete the running job's key"
    )
    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_pipelined_heartbeat_still_restores_identity_on_the_raw_connection(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()
    pipe = FakeRedis()

    worker.heartbeat(pipeline=pipe)

    restored = fake.hashes[worker.key]
    assert restored.get('birth')
    assert restored.get('queues') == 'default'
    assert fake.ttls.get(worker.key), (
        "the recreated key must carry a TTL so a dead worker cannot leave a zombie key behind"
    )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_rebuilds_identity_when_last_heartbeat_attribute_never_set(worker_cls, caplog):
    worker, fake = _make_worker(worker_cls)
    if hasattr(worker, 'last_heartbeat'):
        del worker.last_heartbeat
    fake.hashes[worker.key] = {'last_heartbeat': 'stale', 'successful_job_count': '400'}
    fake.sets['rq:workers'] = set()

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker.heartbeat()

    restored = fake.hashes[worker.key]
    assert restored.get('birth')
    assert restored.get('queues') == 'default'
    assert restored.get('last_heartbeat') == utcformat(worker.last_heartbeat)
    assert fake.ttls.get(worker.key)
    assert worker.key in fake.smembers('rq:workers')
    assert not caplog.records


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_fully_expired_worker_key_is_rebuilt_with_identity_registry_and_ttl(worker_cls):
    worker, fake = _make_worker(worker_cls)

    worker.heartbeat()

    restored = fake.hashes[worker.key]
    assert restored.get('birth')
    assert restored.get('last_heartbeat')
    assert restored.get('queues') == 'default'
    assert fake.ttls.get(worker.key) == worker.worker_ttl + 60
    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_failed_reregistration_cannot_leave_an_expired_key_without_ttl(
    worker_cls, monkeypatch, caplog
):
    worker, fake = _make_worker(worker_cls)

    def fail_registration(*args, **kwargs):
        raise RuntimeError('registration failed')

    monkeypatch.setattr(rhw.worker_registration, 'register', fail_registration)

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        worker.heartbeat()

    assert fake.hashes[worker.key].get('last_heartbeat')
    assert fake.ttls.get(worker.key) == worker.worker_ttl + 60
    assert worker.key not in fake.smembers('rq:workers')
    assert any('re-registration on heartbeat failed' in r.message for r in caplog.records)


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_repair_identity_registration_and_expire_run_through_one_pipeline(worker_cls):
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

    assert ['hset', 'sadd', 'sadd', 'expire'] in seen
    assert fake.ttls.get(worker.key)


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_reraises_stop_requested_for_warm_shutdown(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    def raise_stop(*args, **kwargs):
        raise rhw.StopRequested()

    fake.hexists = raise_stop

    with pytest.raises(rhw.StopRequested):
        worker.heartbeat()


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


def test_beat_skips_the_window_before_execution_is_prepared():
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = None
    job = types.SimpleNamespace(id='job-not-prepared')
    called = []
    worker.maintain_heartbeats = lambda _job: called.append(True)

    worker._refresh_job_heartbeat(job, threading.Event(), threading.Lock())

    assert not called


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


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_override_queues_the_same_command_count_as_rqs_own_heartbeat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'birth': 'ORIGINAL'}
    fake.sets['rq:workers'] = {worker.key}

    ours = FakePipeline(fake)
    worker.heartbeat(timeout=90, pipeline=ours)
    theirs = FakePipeline(fake)
    super(rhw.ReregisterOnHeartbeatMixin, worker).heartbeat(timeout=90, pipeline=theirs)

    assert len(ours.commands) == len(theirs.commands), (
        "the override replaces rq's heartbeat rather than calling it, and rq 2.7.0 reads "
        "maintain_heartbeats results by FIXED index (results[7] = job.heartbeat); if an rq "
        "bump changes how many commands the real heartbeat queues, that index shifts onto "
        "an EXPIRE and rq deletes a RUNNING job's key on every monitor beat"
    )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_never_resurrects_a_worker_that_registered_its_death(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'death': 'YESTERDAY'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    assert worker.key not in fake.smembers('rq:workers'), (
        "register_death unregisters the worker, so a straggler beat that re-registered "
        "it would put a dead worker back on the dashboard"
    )
    assert 'birth' not in fake.hashes[worker.key], (
        "a dead worker's hash must not be rebuilt by a straggler beat; rq's own "
        "heartbeat still refreshes the key TTL, which the mixin cannot prevent"
    )


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
def test_restored_identity_keeps_the_worker_busy_instead_of_reporting_idle(worker_cls):
    worker, fake = _make_worker(worker_cls)
    worker._state = 'busy'
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    assert fake.hashes[worker.key].get('state') == 'busy', (
        "register_birth does not write state, so a rebuilt hash that omits it makes a "
        "worker mid-job advertise as idle on the dashboard"
    )


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_worker_is_advertised_only_after_its_identity_hash_is_rebuilt(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()
    order = []
    real_sadd, real_hset = fake.sadd, fake.hset

    def track_sadd(name, *members):
        order.append('sadd')
        return real_sadd(name, *members)

    def track_hset(name, key=None, value=None, mapping=None):
        if mapping:
            order.append('hset')
        return real_hset(name, key, value, mapping)

    fake.sadd, fake.hset = track_sadd, track_hset

    worker.heartbeat()

    assert order.index('hset') < order.index('sadd'), (
        "advertising the key in rq:workers before the hash exists lets a concurrent "
        "Worker.all() see a member whose hash has no birth and drop it again"
    )
