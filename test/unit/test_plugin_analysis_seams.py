# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the two in-analysis plugin seams.

Covers the extended ``register_onnx_provider`` (per-model scoping) and the newly
wired ``register_analysis_provider`` (component replacement), plus the code paths
that consume them: ``resolve_providers`` in the ONNX session builder and
``get_asr_backend`` in the lyrics pipeline. Everything here runs on CPU with fake
providers and stub backends, so the seams are testable without any GPU.

Main Features:
* register_onnx_provider stores only_models/exclude_models and resolve_providers
  honors them per session label, leaving unmatched models on the default chain
* PluginManager.get_onnx_providers only surfaces providers from loaded plugins
* register_analysis_provider/get_analysis_provider swap a whole component (asr),
  resolving module objects and zero-arg factories and swallowing broken plugins
* lyrics.get_asr_backend prefers a registered override and falls back to built-in
"""

import sys
import types

import pytest

import plugin.api as api
import plugin.manager as manager


def _record(plugin_id, load_status='ok', onnx_providers=None, analysis_providers=None):
    return {
        'id': plugin_id,
        'name': plugin_id,
        'version': '1.0.0',
        'manifest': {},
        'checksum': 'x',
        'requirements': [],
        'enabled': True,
        'settings': {},
        'source_repo': None,
        'load_status': load_status,
        'menu_items': [],
        'cron_tasks': {},
        'onnx_providers': onnx_providers or [],
        'analysis_providers': analysis_providers or {},
        'error': None,
    }


class TestRegisterOnnxProviderScoping:
    def test_stores_scoping_fields(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider(
            'FakeGpuExecutionProvider',
            {'device_id': 0},
            only_models=['musicnn', 'clap'],
        )
        provider = ctx.onnx_providers[0]
        assert provider['name'] == 'FakeGpuExecutionProvider'
        assert provider['options'] == {'device_id': 0}
        assert provider['only_models'] == ['musicnn', 'clap']
        assert provider['exclude_models'] is None

    def test_unscoped_provider_has_none_scopes(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider('FakeGpuExecutionProvider')
        provider = ctx.onnx_providers[0]
        assert provider['only_models'] is None
        assert provider['exclude_models'] is None
        assert provider['options'] == {}

    def test_scopes_are_copied_not_aliased(self):
        ctx = api.PluginContext('demo', 'worker')
        only = ['musicnn']
        ctx.register_onnx_provider('FakeGpuExecutionProvider', only_models=only)
        only.append('clap')
        assert ctx.onnx_providers[0]['only_models'] == ['musicnn']


class TestResolveProvidersScoping:
    """resolve_providers must apply plugin providers only to matching labels."""

    @pytest.fixture
    def song(self, monkeypatch):
        song = pytest.importorskip('tasks.analysis.song')
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: [
                'FakeGpuExecutionProvider', 'CPUExecutionProvider'
            ]
        )
        monkeypatch.setattr(song, 'ort', fake_ort)
        return song

    def _names(self, chain):
        return [name for name, _opts in chain]

    def test_unscoped_provider_applies_to_every_label(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {'device_id': 0},
             'position': 'before_cpu', 'only_models': None, 'exclude_models': None},
        ])
        for label in ('musicnn', 'clap', 'whisper_encoder', None):
            names = self._names(song.resolve_providers(label=label))
            assert names == ['FakeGpuExecutionProvider', 'CPUExecutionProvider']

    def test_only_models_limits_to_matching_label(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': ['musicnn'], 'exclude_models': None},
        ])
        assert 'FakeGpuExecutionProvider' in self._names(song.resolve_providers(label='musicnn'))
        # clap is not in only_models, so it keeps the plain CPU chain.
        assert self._names(song.resolve_providers(label='clap')) == ['CPUExecutionProvider']

    def test_exclude_models_skips_matching_label(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': None,
             'exclude_models': ['whisper_encoder']},
        ])
        assert 'FakeGpuExecutionProvider' in self._names(song.resolve_providers(label='musicnn'))
        assert self._names(
            song.resolve_providers(label='whisper_encoder')
        ) == ['CPUExecutionProvider']

    def test_unavailable_provider_is_dropped(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'MissingExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': None, 'exclude_models': None},
        ])
        assert self._names(song.resolve_providers(label='musicnn')) == ['CPUExecutionProvider']

    def test_cpu_provider_is_always_last(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': None, 'exclude_models': None},
        ])
        assert self._names(song.resolve_providers(label='musicnn'))[-1] == 'CPUExecutionProvider'


class TestManagerOnnxProviders:
    def test_only_loaded_plugins_contribute(self):
        mgr = manager.PluginManager()
        loaded = {'name': 'FakeGpuExecutionProvider', 'options': {}, 'position': 'before_cpu',
                  'only_models': ['musicnn'], 'exclude_models': None}
        pending = {'name': 'OtherExecutionProvider', 'options': {}, 'position': 'before_cpu',
                   'only_models': None, 'exclude_models': None}
        mgr.records = {
            'ok_plugin': _record('ok_plugin', load_status='ok', onnx_providers=[loaded]),
            'unloaded': _record('unloaded', load_status=None, onnx_providers=[pending]),
        }
        providers = mgr.get_onnx_providers()
        assert providers == [loaded]

    def test_deps_failed_status_still_contributes(self):
        # 'deps_failed' is a loaded status: the plugin registered before pip failed.
        mgr = manager.PluginManager()
        provider = {'name': 'FakeGpuExecutionProvider', 'options': {}, 'position': 'before_cpu',
                    'only_models': None, 'exclude_models': None}
        mgr.records = {'p': _record('p', load_status='deps_failed', onnx_providers=[provider])}
        assert mgr.get_onnx_providers() == [provider]


class TestRegisterAnalysisProvider:
    def test_context_stores_factory(self):
        ctx = api.PluginContext('demo', 'worker')
        backend = object()
        ctx.register_analysis_provider('asr', backend)
        assert ctx.analysis_providers == {'asr': backend}

    def test_manager_returns_module_object_directly(self):
        mgr = manager.PluginManager()
        backend = object()
        mgr.records = {'p': _record('p', analysis_providers={'asr': backend})}
        assert mgr.get_analysis_provider('asr') is backend

    def test_manager_resolves_zero_arg_factory(self):
        mgr = manager.PluginManager()
        backend = object()
        mgr.records = {'p': _record('p', analysis_providers={'asr': lambda: backend})}
        assert mgr.get_analysis_provider('asr') is backend

    def test_unregistered_component_returns_none(self):
        mgr = manager.PluginManager()
        mgr.records = {'p': _record('p', analysis_providers={'asr': object()})}
        assert mgr.get_analysis_provider('embedding') is None

    def test_unloaded_plugin_is_not_consulted(self):
        mgr = manager.PluginManager()
        backend = object()
        mgr.records = {'p': _record('p', load_status=None, analysis_providers={'asr': backend})}
        assert mgr.get_analysis_provider('asr') is None

    def test_broken_factory_is_swallowed(self):
        mgr = manager.PluginManager()

        def boom():
            raise RuntimeError('backend import failed')

        mgr.records = {'p': _record('p', analysis_providers={'asr': boom})}
        assert mgr.get_analysis_provider('asr') is None


class TestGetAsrBackend:
    @pytest.fixture
    def asr(self):
        return pytest.importorskip('lyrics._asr_backend')

    def test_override_is_used_when_registered(self, asr, monkeypatch):
        backend = object()
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider',
            lambda component: backend if component == 'asr' else None,
        )
        assert asr.get_asr_backend() is backend

    def test_falls_back_to_builtin_when_no_override(self, asr, monkeypatch):
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider', lambda component: None
        )
        stub = types.ModuleType('lyrics.whisper_onnx')
        monkeypatch.setitem(sys.modules, 'lyrics.whisper_onnx', stub)
        assert asr.get_asr_backend() is stub

    def test_manager_error_falls_back_to_builtin(self, asr, monkeypatch):
        def boom(component):
            raise RuntimeError('plugin manager exploded')

        monkeypatch.setattr(manager.plugin_manager, 'get_analysis_provider', boom)
        stub = types.ModuleType('lyrics.whisper_onnx')
        monkeypatch.setitem(sys.modules, 'lyrics.whisper_onnx', stub)
        assert asr.get_asr_backend() is stub
