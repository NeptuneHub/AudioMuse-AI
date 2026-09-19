# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""HTTP client the end-to-end tests use against the live Flask process.

A thin requests.Session with the task-polling helpers every module needs:
start a queue task (202 + task_id), poll GET /api/status/<id> to a terminal
state, and wait for the queue to be idle before starting the next main task,
because the start endpoints refuse while another main task is live.

Main Features:
* json() asserts the status code and returns the parsed body, quoting the
  response text on a mismatch
* wait_for_task raises TaskOutcomeError carrying the task details when the
  terminal state is not the expected one
* wait_idle polls GET /api/active_tasks until it answers {}
"""

import time

import requests

TERMINAL_STATES = ('SUCCESS', 'FAIL', 'REVOKED')


class TaskOutcomeError(AssertionError):
    pass


class ApiClient:
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def url(self, path):
        return self.base_url + path

    def request(self, method, path, **kwargs):
        kwargs.setdefault('timeout', self.timeout)
        return self.session.request(method, self.url(path), **kwargs)

    def get(self, path, **kwargs):
        return self.request('GET', path, **kwargs)

    def post(self, path, **kwargs):
        return self.request('POST', path, **kwargs)

    def put(self, path, **kwargs):
        return self.request('PUT', path, **kwargs)

    def delete(self, path, **kwargs):
        return self.request('DELETE', path, **kwargs)

    def json(self, method, path, expect=200, **kwargs):
        response = self.request(method, path, **kwargs)
        assert response.status_code == expect, (
            f'{method} {path}: expected {expect}, got {response.status_code}: {response.text[:800]}'
        )
        return response.json()

    def start_task(self, path, json=None):
        body = self.json('POST', path, expect=202, json=json or {})
        task_id = body.get('task_id')
        assert task_id, f'POST {path} answered 202 without a task_id: {body}'
        return task_id

    def status(self, task_id):
        response = self.get(f'/api/status/{task_id}')
        if response.status_code == 404:
            return None
        assert response.status_code == 200, (
            f'GET /api/status/{task_id}: {response.status_code}: {response.text[:400]}'
        )
        return response.json()

    def wait_for_task(self, task_id, timeout, expect='SUCCESS', interval=1.0):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.status(task_id)
            if last is None:
                raise TaskOutcomeError(f'task {task_id} disappeared from task_status while being polled')
            state = last.get('state')
            if state in TERMINAL_STATES:
                if expect is not None and state != expect:
                    raise TaskOutcomeError(
                        f'task {task_id} ({last.get("task_type_from_db")}) ended {state}, expected {expect}: '
                        f'{last.get("status_message")}; details={last.get("details")}'
                    )
                return last
            time.sleep(interval)
        raise TaskOutcomeError(
            f'task {task_id} not terminal after {timeout:.0f}s; last state {last and last.get("state")}: '
            f'{last and last.get("status_message")}'
        )

    def active_task(self):
        return self.json('GET', '/api/active_tasks')

    def wait_idle(self, timeout=180, interval=1.0):
        deadline = time.monotonic() + timeout
        active = self.active_task()
        while active:
            if time.monotonic() >= deadline:
                raise TaskOutcomeError(f'queue still busy after {timeout:.0f}s: {active}')
            time.sleep(interval)
            active = self.active_task()
        return True

    def last_task(self):
        return self.json('GET', '/api/last_task')

    def health(self):
        response = self.get('/api/health', timeout=5)
        return response.status_code == 200 and response.json().get('status') == 'ok'
