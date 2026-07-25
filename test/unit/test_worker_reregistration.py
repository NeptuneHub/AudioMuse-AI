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
import time
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

    def expire(self, *args, **kwargs):
        self.commands.append(('expire', args, kwargs))

    def execute(self):
        results = [getattr(self.parent, name)(*args, **kwargs) for name, args, kwargs in self.commands]
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
    pipe = FakeRedis()

    worker.heartbeat(pipeline=pipe)

    assert pipe.sets == {}, (
        "rq 2.7.0 reads maintain_heartbeats pipeline results by FIXED index "
        "(results[7] = job.heartbeat); an injected command would shift it onto an "
        "EXPIRE and delete the running job's key every monitor beat"
    )
    assert set(pipe.hashes.get(worker.key, {})) <= {'last_heartbeat'}
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
        "the recreated key must carry a TTL so a dead worker cannot leave a "
        "zombie key behind"
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
    assert fake.ttls.get(worker.key)
    assert worker.key in fake.smembers('rq:workers')
    assert not caplog.records


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_repair_hset_and_expire_run_through_one_pipeline(worker_cls):
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

    assert ['hset', 'expire'] in seen
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


def _drive_one_beat(worker, job, monkeypatch, maintain):
    worker.job_monitoring_interval = 1
    worker.maintain_heartbeats = maintain
    monkeypatch.setattr(
        rhw.SimpleWorker, 'execute_job', lambda self, j, q: time.sleep(1.6)
    )
    worker.execute_job(job, None)


def test_beat_thread_stays_silent_when_execution_is_cleared_mid_refresh(caplog, monkeypatch):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = object()
    job = types.SimpleNamespace(id='job-teardown')

    def clear_then_fail(_job):
        worker.execution = None
        raise AttributeError("'NoneType' object has no attribute 'heartbeat'")

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        _drive_one_beat(worker, job, monkeypatch, clear_then_fail)

    assert not caplog.records, (
        "cleanup_execution nulls worker.execution while the beat thread is already "
        "inside maintain_heartbeats, so the teardown race must exit quietly instead "
        "of logging an ERROR traceback on every finished job"
    )


def test_beat_thread_still_logs_a_genuine_heartbeat_failure(caplog, monkeypatch):
    worker, _ = _make_worker(rhw.HeartbeatSimpleWorker)
    worker.execution = object()
    job = types.SimpleNamespace(id='job-real-failure')

    def fail(_job):
        raise OSError('redis is down')

    with caplog.at_level(logging.ERROR, logger='rq_heartbeat_worker'):
        _drive_one_beat(worker, job, monkeypatch, fail)

    assert any('Heartbeat refresh failed' in r.message for r in caplog.records), (
        "silencing the teardown race must not silence a real Redis outage"
    )
