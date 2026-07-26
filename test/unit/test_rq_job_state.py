# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the centralized RQ job-state classifier.

Locks the invariant that made this module necessary: every status RQ can report
must land in exactly one bucket. A status in neither bucket used to be judged
'dead' by the sweep recovery and 'unknown, leave alone' by the orphan reaper, so
the same job drove opposite actions.

Main Features:
* Total coverage of rq's JobStatus enum, so an rq upgrade that adds a status
  fails here instead of silently stranding a task at "running".
* status_value normalizes enum, str, bytes, 'JobStatus.X' reprs and None.
* A 'started' job is abandoned only once its own heartbeat goes stale, and a
  missing heartbeat is never read as death.
"""

import datetime

import pytest

import rq_job_state


def test_every_rq_job_status_is_either_alive_or_terminal():
    from rq.job import JobStatus

    unclassified = [
        member.value for member in JobStatus
        if not rq_job_state.is_alive_status(member)
        and not rq_job_state.is_terminal_status(member)
    ]
    assert unclassified == []


def test_no_rq_status_is_both_alive_and_terminal():
    from rq.job import JobStatus

    both = [
        member.value for member in JobStatus
        if rq_job_state.is_alive_status(member)
        and rq_job_state.is_terminal_status(member)
    ]
    assert both == []


def test_a_created_but_not_yet_enqueued_job_counts_as_alive_not_dead():
    assert rq_job_state.is_alive_status('created') is True
    assert rq_job_state.is_terminal_status('created') is False


def test_every_cancelled_status_is_also_terminal():
    for status in ('canceled', 'stopped'):
        assert rq_job_state.is_cancelled_status(status) is True
        assert rq_job_state.is_terminal_status(status) is True


def test_a_failed_job_is_terminal_but_not_a_cancellation():
    assert rq_job_state.is_terminal_status('failed') is True
    assert rq_job_state.is_cancelled_status('failed') is False


@pytest.mark.parametrize('raw', ['queued', 'QUEUED', ' Queued ', b'queued'])
def test_status_value_normalizes_case_padding_and_bytes(raw):
    assert rq_job_state.status_value(raw) == 'queued'


def test_status_value_strips_the_enum_repr_prefix():
    assert rq_job_state.status_value('JobStatus.QUEUED') == 'queued'


def test_status_value_of_a_real_enum_member_matches_its_value():
    from rq.job import JobStatus

    assert rq_job_state.status_value(JobStatus.STARTED) == 'started'


def test_status_value_of_none_is_empty_and_classified_as_neither():
    assert rq_job_state.status_value(None) == ''
    assert rq_job_state.is_alive_status(None) is False
    assert rq_job_state.is_terminal_status(None) is False


class _Job:
    def __init__(self, status, last_heartbeat=None):
        self._status = status
        self.last_heartbeat = last_heartbeat

    def get_status(self, refresh=False):
        return self._status


def _state(status, last_heartbeat=None, abandoned_after=300):
    state, _value = rq_job_state._classify(
        status, job=_Job(status, last_heartbeat), abandoned_after=abandoned_after
    )
    return state


def test_a_started_job_with_a_stale_heartbeat_is_abandoned_not_terminal():
    stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    state = _state('started', last_heartbeat=stale)
    assert rq_job_state.is_abandoned(state) is True
    assert rq_job_state.is_terminal(state) is False


def test_a_started_job_with_a_fresh_heartbeat_stays_alive():
    fresh = datetime.datetime.now(datetime.timezone.utc)
    assert rq_job_state.is_alive(_state('started', last_heartbeat=fresh)) is True


def test_a_naive_heartbeat_timestamp_is_read_as_utc_not_as_an_error():
    stale_naive = datetime.datetime.now(datetime.timezone.utc).replace(
        tzinfo=None
    ) - datetime.timedelta(hours=1)
    assert rq_job_state.is_abandoned(_state('started', last_heartbeat=stale_naive)) is True


def test_a_started_job_with_no_heartbeat_at_all_is_left_alive():
    assert rq_job_state.is_alive(_state('started', last_heartbeat=None)) is True


def test_a_zero_abandoned_window_disables_the_heartbeat_check():
    stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    state = _state('started', last_heartbeat=stale, abandoned_after=0)
    assert rq_job_state.is_alive(state) is True


def test_only_started_jobs_are_ever_judged_abandoned():
    stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    for status in ('queued', 'deferred', 'scheduled', 'created'):
        assert rq_job_state.is_alive(_state(status, last_heartbeat=stale)) is True


def test_an_unrecognized_status_is_unknown_so_callers_leave_the_row_alone():
    state, value = rq_job_state._classify('some-future-rq-status', job=None)
    assert rq_job_state.is_unknown(state) is True
    assert value == 'some-future-rq-status'


def test_the_state_verdicts_are_never_mistaken_for_rq_statuses():
    for verdict in ('alive', 'terminal', 'abandoned', 'missing', 'unknown'):
        assert rq_job_state.is_alive_status(verdict) is False
        assert rq_job_state.is_terminal_status(verdict) is False


class _RetryJob:
    def __init__(self, retries_left, last_heartbeat=None, status='started'):
        self.retries_left = retries_left
        self.last_heartbeat = last_heartbeat
        self.saved = 0
        self._status = status

    def get_status(self, refresh=False):
        return self._status

    def save(self, **kwargs):
        self.saved += 1


def _stale_heartbeat():
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)


def test_forbid_retries_zeroes_the_budget_so_rq_cleanup_cannot_requeue_it(monkeypatch):
    import rq.job

    job = _RetryJob(3)
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )

    assert rq_job_state.forbid_retries('cancelled-job', object()) is True
    assert job.retries_left == 0
    assert job.saved == 1


def test_forbid_retries_leaves_a_job_that_never_had_retries_untouched(monkeypatch):
    import rq.job

    job = _RetryJob(None)
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )

    assert rq_job_state.forbid_retries('plain-job', object()) is False
    assert job.saved == 0


def test_forbid_retries_survives_a_job_that_is_already_gone(monkeypatch):
    import rq.exceptions
    import rq.job

    def missing(job_id, connection=None):
        raise rq.exceptions.NoSuchJobError('gone')

    monkeypatch.setattr(rq.job.Job, 'fetch', staticmethod(missing))

    assert rq_job_state.forbid_retries('gone-job', object()) is False


def test_a_zeroed_budget_makes_retry_abandoned_job_answer_no_budget(monkeypatch):
    import rq.job

    job = _RetryJob(0, last_heartbeat=_stale_heartbeat())
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )

    verdict = rq_job_state.retry_abandoned_job('cancelled-job', object())
    assert verdict == rq_job_state.RETRY_NO_BUDGET


def test_a_job_that_recovered_between_probe_and_retry_is_not_requeued_or_failed(
    monkeypatch,
):
    import rq.job

    fresh = datetime.datetime.now(datetime.timezone.utc)
    job = _RetryJob(3, last_heartbeat=fresh)
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )

    verdict = rq_job_state.retry_abandoned_job('woken-job', object())
    assert verdict == rq_job_state.RETRY_RECOVERED


def test_a_job_that_finished_between_probe_and_retry_is_not_requeued(monkeypatch):
    import rq.job

    job = _RetryJob(3, status='finished')
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )

    verdict = rq_job_state.retry_abandoned_job('done-job', object())
    assert verdict == rq_job_state.RETRY_RECOVERED


def test_a_vanished_job_answers_missing_not_out_of_retries(monkeypatch):
    import rq.exceptions
    import rq.job

    def gone(job_id, connection=None):
        raise rq.exceptions.NoSuchJobError('gone')

    monkeypatch.setattr(rq.job.Job, 'fetch', staticmethod(gone))

    verdict = rq_job_state.retry_abandoned_job('gone-job', object())
    assert verdict == rq_job_state.RETRY_MISSING


def test_a_probe_error_answers_error_so_the_row_is_left_for_the_next_pass(monkeypatch):
    import rq.job

    def boom(job_id, connection=None):
        raise ConnectionError('redis blinked')

    monkeypatch.setattr(rq.job.Job, 'fetch', staticmethod(boom))

    verdict = rq_job_state.retry_abandoned_job('blinked-job', object())
    assert verdict == rq_job_state.RETRY_ERROR


def test_requeueing_drops_the_dead_started_registry_entry_first(monkeypatch):
    from unittest.mock import MagicMock

    import rq.job

    job = MagicMock()
    job.id = 'abandoned-1'
    job.get_status.return_value = 'started'
    job.last_heartbeat = _stale_heartbeat()
    job.retries_left = 2
    job.origin = 'high'
    job.started_job_registry.key = 'rq:wip:high'
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda job_id, connection=None: job)
    )
    monkeypatch.setattr(
        rq_job_state, 'Queue', lambda name, connection=None: ('queue', name)
    )
    connection = MagicMock()
    connection.zrange.return_value = [b'abandoned-1:exec-9', b'other-job:exec-1']

    verdict = rq_job_state.retry_abandoned_job('abandoned-1', connection)

    assert verdict == rq_job_state.RETRY_REQUEUED
    connection.zrem.assert_called_once_with('rq:wip:high', 'abandoned-1:exec-9')
    assert job.retry.call_count == 1


def test_probe_jobs_many_maps_absent_jobs_to_missing_and_classifies_the_rest(
    monkeypatch,
):
    class _BatchJob:
        @staticmethod
        def fetch_many(job_ids, connection=None):
            return [_RetryJob(1, status='finished'), None]

    monkeypatch.setattr(rq_job_state, 'Job', _BatchJob)

    results = rq_job_state.probe_jobs_many(['done', 'gone'], object())
    assert rq_job_state.is_terminal(results['done'][0])
    assert results['done'][1] == 'finished'
    assert rq_job_state.is_missing(results['gone'][0])


def test_probe_jobs_many_falls_back_to_single_probes_when_the_batch_fetch_fails(
    monkeypatch,
):
    class _BrokenBatchJob:
        @staticmethod
        def fetch_many(job_ids, connection=None):
            raise ConnectionError('pipeline refused')

        @staticmethod
        def fetch(job_id, connection=None):
            return _RetryJob(1, status='queued')

    monkeypatch.setattr(rq_job_state, 'Job', _BrokenBatchJob)

    results = rq_job_state.probe_jobs_many(['q1', 'q2'], object())
    assert all(rq_job_state.is_alive(state) for state, _value in results.values())
