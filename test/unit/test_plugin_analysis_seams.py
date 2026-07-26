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
* register_onnx_provider stores only_models/exclude_models (a single label may be
  a plain string), position and needs_static_shapes, and resolve_providers honors
  them per session label, leaving unmatched models on the default chain
* Sessions core keeps on CPU (gte, silero_vad) stay on CPU unless a plugin opts in
  by naming them in only_models
* PluginManager.get_onnx_providers only surfaces providers from loaded plugins
* register_analysis_provider/get_analysis_provider swap a whole component (asr),
  resolving module objects and zero-arg factories once (unless cache=False),
  rejecting unknown components and swallowing broken plugins
* lyrics.get_asr_backend prefers a registered override, and falls back to the
  built-in when there is none or the override misses part of the whisper surface
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

    def test_single_label_may_be_a_plain_string(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider(
            'FakeGpuExecutionProvider', only_models='musicnn', exclude_models='clap'
        )
        provider = ctx.onnx_providers[0]
        assert provider['only_models'] == ['musicnn']
        assert provider['exclude_models'] == ['clap']

    def test_unknown_position_falls_back_to_before_cpu(self, caplog):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider('FakeGpuExecutionProvider', position='after_everything')
        assert ctx.onnx_providers[0]['position'] == 'before_cpu'
        assert 'unknown position' in caplog.text

    def test_static_shapes_flag_is_stored(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider('FakeGpuExecutionProvider', needs_static_shapes=True)
        assert ctx.onnx_providers[0]['needs_static_shapes'] is True

    def test_static_shapes_defaults_to_false(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_onnx_provider('FakeGpuExecutionProvider')
        assert ctx.onnx_providers[0]['needs_static_shapes'] is False


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

    @pytest.fixture
    def song_with_cuda(self, song, monkeypatch):
        monkeypatch.setattr(song, 'ort', types.SimpleNamespace(
            get_available_providers=lambda: [
                'CUDAExecutionProvider', 'FakeGpuExecutionProvider', 'CPUExecutionProvider'
            ]
        ))
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

    def test_string_scope_is_accepted(self, song, monkeypatch):
        # A provider stored by an older plugin (or by hand) may carry a bare string.
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': 'musicnn', 'exclude_models': None},
        ])
        assert 'FakeGpuExecutionProvider' in self._names(song.resolve_providers(label='musicnn'))
        assert self._names(song.resolve_providers(label='clap')) == ['CPUExecutionProvider']

    def test_unknown_label_is_reported(self, song, monkeypatch, caplog):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': ['musicnnn'], 'exclude_models': None},
        ])
        song.resolve_providers(label='musicnn')
        assert 'unknown only_models' in caplog.text

    def test_known_labels_are_not_reported(self, song, monkeypatch, caplog):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {}, 'position': 'before_cpu',
             'only_models': sorted(song.MODEL_LABELS), 'exclude_models': None},
        ])
        song.resolve_providers(label='musicnn')
        assert 'unknown' not in caplog.text

    def test_before_cuda_position_wins_over_cuda(self, song_with_cuda, monkeypatch):
        monkeypatch.setattr(song_with_cuda, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cuda', 'only_models': None, 'exclude_models': None},
        ])
        assert self._names(song_with_cuda.resolve_providers(label='musicnn')) == [
            'FakeGpuExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        ]

    def test_before_cpu_position_follows_cuda(self, song_with_cuda, monkeypatch):
        monkeypatch.setattr(song_with_cuda, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': None, 'exclude_models': None},
        ])
        assert self._names(song_with_cuda.resolve_providers(label='musicnn')) == [
            'CUDAExecutionProvider', 'FakeGpuExecutionProvider', 'CPUExecutionProvider'
        ]


class TestCpuOnlyDefaultSessions:
    """Sessions core keeps on CPU stay there unless a plugin asks for them."""

    @pytest.fixture
    def song(self, monkeypatch):
        song = pytest.importorskip('tasks.analysis.song')
        monkeypatch.setattr(song, 'ort', types.SimpleNamespace(
            get_available_providers=lambda: [
                'CUDAExecutionProvider', 'FakeGpuExecutionProvider', 'CPUExecutionProvider'
            ]
        ))
        return song

    def _names(self, chain):
        return [name for name, _opts in chain]

    def test_builtin_gpu_providers_are_skipped(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [])
        chain = song.resolve_providers(label='gte', cpu_only_default=True)
        assert self._names(chain) == ['CPUExecutionProvider']

    def test_unscoped_plugin_provider_is_not_used(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': None, 'exclude_models': None},
        ])
        chain = song.resolve_providers(label='gte', cpu_only_default=True)
        assert self._names(chain) == ['CPUExecutionProvider']

    def test_plugin_provider_can_opt_in_by_label(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': ['gte'], 'exclude_models': None},
        ])
        chain = song.resolve_providers(label='gte', cpu_only_default=True)
        assert self._names(chain) == ['FakeGpuExecutionProvider', 'CPUExecutionProvider']

    def test_opt_in_for_another_label_does_not_leak(self, song, monkeypatch):
        monkeypatch.setattr(song, '_plugin_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'options': {},
             'position': 'before_cpu', 'only_models': ['gte'], 'exclude_models': None},
        ])
        chain = song.resolve_providers(label='silero_vad', cpu_only_default=True)
        assert self._names(chain) == ['CPUExecutionProvider']


class TestStaticShapeProviders:
    """CLAP must learn which providers need pinned shapes from the registration."""

    @pytest.fixture
    def clap(self):
        return pytest.importorskip('tasks.clap_analyzer')

    def test_plugin_provider_opts_in(self, clap, monkeypatch):
        monkeypatch.setattr(manager.plugin_manager, 'get_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'needs_static_shapes': True},
        ])
        assert 'FakeGpuExecutionProvider' in clap._static_shape_providers()

    def test_provider_without_the_flag_is_left_out(self, clap, monkeypatch):
        monkeypatch.setattr(manager.plugin_manager, 'get_onnx_providers', lambda: [
            {'name': 'FakeGpuExecutionProvider', 'needs_static_shapes': False},
        ])
        assert 'FakeGpuExecutionProvider' not in clap._static_shape_providers()

    def test_builtin_coreml_is_always_there(self, clap, monkeypatch):
        monkeypatch.setattr(manager.plugin_manager, 'get_onnx_providers', lambda: [])
        assert 'CoreMLExecutionProvider' in clap._static_shape_providers()

    def test_broken_plugin_manager_is_survivable(self, clap, monkeypatch):
        def boom():
            raise RuntimeError('plugin manager exploded')

        monkeypatch.setattr(manager.plugin_manager, 'get_onnx_providers', boom)
        assert clap._static_shape_providers() == {'CoreMLExecutionProvider'}


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
        assert ctx.analysis_providers == {'asr': {'factory': backend, 'cache': True}}

    def test_cache_opt_out_is_stored(self):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_analysis_provider('asr', object(), cache=False)
        assert ctx.analysis_providers['asr']['cache'] is False

    def test_unknown_component_is_ignored(self, caplog):
        ctx = api.PluginContext('demo', 'worker')
        ctx.register_analysis_provider('ASR', object())
        assert ctx.analysis_providers == {}
        assert 'unknown component' in caplog.text

    def test_double_registration_warns_and_keeps_last(self, caplog):
        ctx = api.PluginContext('demo', 'worker')
        first, second = object(), object()
        ctx.register_analysis_provider('asr', first)
        ctx.register_analysis_provider('asr', second)
        assert ctx.analysis_providers['asr']['factory'] is second
        assert 'two analysis providers' in caplog.text

    def test_two_plugins_for_one_component_warn(self, caplog):
        mgr = manager.PluginManager()
        first, second = object(), object()
        mgr.records = {
            'a': _record('a', analysis_providers={'asr': first}),
            'b': _record('b', analysis_providers={'asr': second}),
        }
        assert mgr.get_analysis_provider('asr') is first
        assert 'all provide the analysis component' in caplog.text

    def test_cached_factory_runs_once(self):
        mgr = manager.PluginManager()
        calls = []

        def factory():
            calls.append(1)
            return object()

        mgr.records = {'p': _record('p', analysis_providers={
            'asr': {'factory': factory, 'cache': True},
        })}
        assert mgr.get_analysis_provider('asr') is mgr.get_analysis_provider('asr')
        assert len(calls) == 1

    def test_uncached_factory_runs_every_time(self):
        mgr = manager.PluginManager()
        mgr.records = {'p': _record('p', analysis_providers={
            'asr': {'factory': object, 'cache': False},
        })}
        assert mgr.get_analysis_provider('asr') is not mgr.get_analysis_provider('asr')

    def test_sync_clears_the_cache(self, monkeypatch):
        mgr = manager.PluginManager()
        mgr._analysis_provider_cache = {'asr': object()}
        monkeypatch.setattr(mgr, 'enabled', lambda: False)
        mgr.sync()
        assert mgr._analysis_provider_cache == {}

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

    def _backend(self, **overrides):
        """A stand-in with the full whisper surface, minus anything overridden away."""
        surface = {name: (lambda *a, **k: None) for name in ('load_whisper_model',
                                                             'transcribe', 'is_loaded', 'unload')}
        surface.update(overrides)
        return types.SimpleNamespace(**{k: v for k, v in surface.items() if v is not None})

    def _stub_builtin(self, monkeypatch):
        """Replace the built-in whisper backend with an empty module.

        ``from . import whisper_onnx`` reads the attribute off the package once
        the submodule has been imported by anything else in the session, so
        patching sys.modules alone is not enough.
        """
        stub = types.ModuleType('lyrics.whisper_onnx')
        lyrics_pkg = pytest.importorskip('lyrics')
        monkeypatch.setitem(sys.modules, 'lyrics.whisper_onnx', stub)
        monkeypatch.setattr(lyrics_pkg, 'whisper_onnx', stub, raising=False)
        return stub

    def test_override_is_used_when_registered(self, asr, monkeypatch):
        backend = self._backend()
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider',
            lambda component: backend if component == 'asr' else None,
        )
        assert asr.get_asr_backend() is backend

    @pytest.mark.parametrize(
        'missing', ['load_whisper_model', 'transcribe', 'is_loaded', 'unload']
    )
    def test_incomplete_override_falls_back_to_builtin(self, asr, monkeypatch, caplog, missing):
        backend = self._backend(**{missing: None})
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider', lambda component: backend
        )
        stub = self._stub_builtin(monkeypatch)
        assert asr.get_asr_backend() is stub
        assert missing in caplog.text

    def test_non_callable_attribute_is_not_enough(self, asr, monkeypatch):
        backend = self._backend(is_loaded=True)
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider', lambda component: backend
        )
        stub = self._stub_builtin(monkeypatch)
        assert asr.get_asr_backend() is stub

    def test_falls_back_to_builtin_when_no_override(self, asr, monkeypatch):
        monkeypatch.setattr(
            manager.plugin_manager, 'get_analysis_provider', lambda component: None
        )
        stub = self._stub_builtin(monkeypatch)
        assert asr.get_asr_backend() is stub

    def test_manager_error_falls_back_to_builtin(self, asr, monkeypatch):
        def boom(component):
            raise RuntimeError('plugin manager exploded')

        monkeypatch.setattr(manager.plugin_manager, 'get_analysis_provider', boom)
        stub = self._stub_builtin(monkeypatch)
        assert asr.get_asr_backend() is stub
