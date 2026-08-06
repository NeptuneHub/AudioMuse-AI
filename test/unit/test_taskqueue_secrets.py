# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""API keys never become database content, and never appear where they were not sent.

The enqueue strips the secret kwargs out of the payload and the claim restores
them from the worker's own config. Restoring has to be scoped to what was
actually stripped: injecting every configured secret into every claimed job
handed ``openai_api_key_param`` to run_analysis_task, every clustering batch and
the cleaning task - none of which accept it - and each died on TypeError the
moment it started.

Main Features:
* A payload never contains a secret kwarg's value
* A job enqueued with secrets gets exactly those secrets back at claim time
* A job enqueued without secrets has nothing injected into it
* A stripped secret is restored even when this worker's config value is None
"""

import json

import config
from taskqueue import sql


class _Cursor:
    def __init__(self, claim_row=None):
        self.claim_row = claim_row
        self.executed = []

    def execute(self, statement, params=None):
        self.executed.append((' '.join(statement.split()), params))

    def fetchone(self):
        return self.claim_row


SECRET = config.QUEUE_SECRET_KWARGS[0]


def _payload_of(cur):
    return json.loads(cur.executed[0][1][5])


class TestSecretsNeverReachTheDatabase:
    def test_the_payload_holds_the_stripped_names_but_not_their_values(self):
        cur = _Cursor(claim_row=('t-1',))
        sql.insert_job(
            cur, 't-1', 'main_clustering', 'tasks.clustering.run_clustering_task',
            kwargs={SECRET: 'sk-live-abc', 'other': 1},
        )

        payload = _payload_of(cur)
        assert payload['stripped'] == [SECRET]
        assert SECRET not in payload['kwargs']
        assert 'sk-live-abc' not in json.dumps(payload)
        assert payload['kwargs']['other'] == 1


class TestOnlyWhatWasStrippedComesBack:
    def _claimed(self, payload, monkeypatch, config_value='from-config'):
        monkeypatch.setattr(
            config, SECRET[:-len('_param')].upper(), config_value, raising=False
        )
        row = ('t-1', 'album_analysis', None, 'f', json.dumps(payload), 1, 3)
        return sql.claim(_Cursor(claim_row=row), 'default', 0.0)

    def test_a_job_enqueued_without_secrets_has_nothing_injected(self, monkeypatch):
        job = self._claimed({'args': [], 'kwargs': {'top_n': 5}}, monkeypatch)

        assert job['kwargs'] == {'top_n': 5}

    def test_a_stripped_secret_is_restored_from_this_workers_config(self, monkeypatch):
        job = self._claimed(
            {'args': [], 'kwargs': {}, 'stripped': [SECRET]}, monkeypatch
        )

        assert job['kwargs'][SECRET] == 'from-config'

    def test_a_stripped_secret_is_restored_even_when_unconfigured(self, monkeypatch):
        job = self._claimed(
            {'args': [], 'kwargs': {}, 'stripped': [SECRET]},
            monkeypatch, config_value=None,
        )

        assert SECRET in job['kwargs']
        assert job['kwargs'][SECRET] is None

    def test_a_legacy_payload_without_the_stripped_list_gets_no_injection(
        self, monkeypatch
    ):
        job = self._claimed({'args': [], 'kwargs': {'x': 1}}, monkeypatch)

        assert job['kwargs'] == {'x': 1}
