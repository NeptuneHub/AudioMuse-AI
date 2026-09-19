# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Boot order and teardown of the whole end-to-end stack.

One object owns every piece the session fixture hands to the tests: the run
and data directories, the Postgres handle, the model paths, two Navidrome
instances serving the same library (the default server and a second one for
the multi-server and provider-migration scenarios), the gunicorn web process,
the two queue workers, the control listener plus the harness's control server
that stops and starts these processes when the app asks, the fixture library
bound to Navidrome's song ids, the seed catalogue and the HTTP client.

Main Features:
* Stack.boot() raises StackError with the log tail of whichever step failed,
  after which shutdown() is still safe to call
* the seed builder reuses the same boot against another music folder
* shutdown() stops every process, puts moved library files back and cleans
  the ephemeral Postgres
* alive_problems() and summary() feed the per-test liveness check and the
  terminal summary
"""

import json
import os

from . import binaries, library, models, paths, ports, postgres, seed
from .api import ApiClient
from .control import (
    SERVICE_FLASK,
    SERVICE_LISTENER,
    SERVICE_WORKER_DEFAULT,
    SERVICE_WORKER_HIGH,
    ControlServer,
)
from .env import build_env, masked
from .errors import StackError
from .flask import FLASK_PORT, FlaskServer
from .fpcalc import FPCALC_ASSET
from .navidrome import DEFAULT_PORT, NAVIDROME_ASSET, NavidromeServer
from .workers import ControlListener, QueueWorker

QUEUES = ('high', 'default')
SERVICE_QUEUE = {SERVICE_WORKER_HIGH: 'high', SERVICE_WORKER_DEFAULT: 'default'}


class Stack:
    def __init__(self, music_folder=None, load_manifest=True, second_instance=True, load_seed=True):
        self.music_folder = music_folder or paths.LIBRARY_DIR
        self.load_manifest = load_manifest
        self.second_instance = second_instance
        self.load_seed = load_seed
        self.run_dir = None
        self.data_root = None
        self.reused = False
        self.pg = None
        self.model_env = None
        self.fpcalc = None
        self.navidrome = None
        self.navidrome2 = None
        self.flask = None
        self.workers = {}
        self.listener = None
        self.control = None
        self.library = None
        self.seed = None
        self.api = None
        self.env_flask = None
        self.env_worker = None

    @property
    def base_url(self):
        return self.flask.base_url

    @property
    def dsn(self):
        return self.pg.dsn

    @property
    def parts(self):
        return self.pg.parts

    @property
    def subsonic(self):
        return self.navidrome.client

    @property
    def subsonic2(self):
        return self.navidrome2.client if self.navidrome2 is not None else None

    @property
    def expected_files(self):
        manifest_files = self.library.counts['files'] if self.library is not None else 0
        return manifest_files + (self.seed.count if self.seed is not None else 0)

    @property
    def seed_count(self):
        return self.seed.count if self.seed is not None else 0

    @property
    def catalogue_rows(self):
        return self.library.counts['catalogue_rows'] + self.seed_count

    @property
    def analyzable_files(self):
        return self.library.counts['analyzable_files'] + self.seed_count

    def _write_env_json(self):
        payload = {
            'run_dir': self.run_dir,
            'data_root': self.data_root,
            'reused_state': self.reused,
            'music_folder': self.music_folder,
            'seed_tracks': self.seed_count,
            'flask': masked(self.env_flask),
            'worker': masked(self.env_worker),
        }
        with open(os.path.join(self.run_dir, 'env.json'), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _resolve_service(self, service):
        if service == SERVICE_FLASK:
            return self.flask
        if service == SERVICE_LISTENER:
            return self.listener
        queue = SERVICE_QUEUE.get(service)
        return self.workers.get(queue) if queue else None

    def boot(self):
        self.run_dir = paths.prepare_run_dir()
        self.data_root, self.reused = paths.prepare_data_dir()
        if self.load_manifest:
            self.library = library.load()
            self.library.holding_dir = os.path.join(self.data_root, 'holding')
        self.seed = seed.load() if self.load_seed else seed.empty()
        if self.load_seed:
            self.seed.ensure_placeholders(self.music_folder)
        self.pg = postgres.resolve(self.data_root)
        if self.pg is None:
            return False
        if not self.reused:
            postgres.reset_public_schema(self.pg.dsn)
        self.model_env = models.resolve()
        navidrome_bin = binaries.ensure(NAVIDROME_ASSET)
        self.fpcalc = binaries.ensure(FPCALC_ASSET)
        ports.require_free(FLASK_PORT, 'Flask')
        nd_port = DEFAULT_PORT if ports.is_free('127.0.0.1', DEFAULT_PORT) else ports.free_port()
        self.navidrome = NavidromeServer(
            navidrome_bin, self.data_root, self.music_folder, nd_port,
            paths.log_path('navidrome'), os.environ, instance='navidrome',
        )
        if self.second_instance:
            self.navidrome2 = NavidromeServer(
                navidrome_bin, self.data_root, self.music_folder, ports.free_port(),
                paths.log_path('navidrome2'), os.environ, instance='navidrome2',
            )
        control_socket = os.path.join(self.data_root, 'control.sock')
        self.control = ControlServer(control_socket, self._resolve_service, paths.log_path('control'))
        self.control.start()
        extra_path = [self.pg.bin_dir] if self.pg.bin_dir else []
        common = dict(
            pg_parts=self.pg.parts, data_root=self.data_root,
            navidrome_url=self.navidrome.base_url, models=self.model_env,
            fpcalc=self.fpcalc, extra_path=extra_path, control_socket=control_socket,
        )
        self.env_flask = build_env('flask', **common)
        self.env_worker = build_env('worker', **common)
        self._write_env_json()
        self.navidrome.start()
        if self.navidrome2 is not None:
            self.navidrome2.start()
        self.flask = FlaskServer(self.env_flask, paths.log_path('flask'))
        self.flask.start()
        expected = self.expected_files if self.load_manifest else None
        self.navidrome.wait_ready(60)
        self.navidrome.wait_scanned(expected, 180)
        if self.navidrome2 is not None:
            self.navidrome2.wait_ready(60)
            self.navidrome2.wait_scanned(expected, 180)
        if self.library is not None:
            self.library.bind_provider_ids(self.subsonic.all_songs())
        self.flask.wait_ready(240)
        self.api = ApiClient(self.flask.base_url)
        for queue in QUEUES:
            worker = QueueWorker(queue, self.env_worker, paths.log_path(f'worker-{queue}'))
            worker.start()
            self.workers[queue] = worker
        self.listener = ControlListener(self.env_worker, paths.log_path('control-listener'))
        self.listener.start()
        for worker in self.workers.values():
            worker.wait_ready(180)
        self.listener.wait_ready(120)
        return True

    def shutdown(self):
        if self.control is not None:
            self.control.stop()
        if self.listener is not None:
            self.listener.stop()
        for worker in self.workers.values():
            worker.stop()
        if self.flask is not None:
            self.flask.stop()
        for instance in (self.navidrome, self.navidrome2):
            if instance is not None:
                instance.stop()
        if self.library is not None:
            self.library.restore_all()
        if self.pg is not None:
            self.pg.cleanup()

    def alive_problems(self):
        problems = []
        if self.flask is not None and not self.flask.running():
            problems.append(self.flask.process.describe())
        for instance in (self.navidrome, self.navidrome2):
            if instance is not None and not instance.running():
                problems.append(instance.process.describe())
        for worker in self.workers.values():
            if worker.dead:
                problems.append(f'worker {worker.queue} gave up after {worker.restarts} respawns. ' + worker.process.describe())
        if self.listener is not None and self.listener.dead:
            problems.append('the control listener gave up. ' + self.listener.process.describe())
        return problems

    def summary(self):
        lines = [f'e2e run dir: {self.run_dir}', f'e2e data root: {self.data_root}', f'seed tracks: {self.seed_count}']
        for worker in self.workers.values():
            lines.append(f'worker {worker.queue}: {worker.restarts} respawn(s)')
        if self.control is not None:
            lines.append(f'control requests served: {len(self.control.requests)}')
        return '\n'.join(lines)

    def rescan_library(self, expected_files=None, timeout=180):
        expected = self.expected_files if expected_files is None else expected_files
        for instance in (self.navidrome, self.navidrome2):
            if instance is not None:
                instance.rescan(expected, timeout)
        if self.library is not None and expected == self.expected_files:
            self.library.bind_provider_ids(self.subsonic.all_songs())


def require_linux():
    if os.name != 'posix':
        raise StackError('the end-to-end stack is Linux only (gunicorn, fork-per-job workers): run it through WSL')
