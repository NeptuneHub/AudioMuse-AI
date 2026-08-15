# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Container-aware CPU count resolution in cpu_budget.

Covers the cgroup v1 and v2 quota readers, the affinity probe, and the
guarantee that detect_cpu_count can only ever return the caller's existing
fallback when anything at all looks wrong.

Main Features:
* cgroup v2 cpu.max and cgroup v1 quota/period pairs resolve to whole CPUs
* Unlimited, malformed, zero and negative quotas resolve to no quota
* The smallest of quota, affinity and host count wins
* With no restriction below the host count, ONNX is left at its own default
* Any probe explosion, or a nonsensical value, returns the caller's fallback
* A real but below-minimum reading is raised to the minimum, never discarded
* The module imports no numeric library, which the worker entrypoints rely on
* The worker, lyrics and secondary-pool call sites size from the same probe
"""

import importlib
import logging
import os
import subprocess
import sys

import pytest

import cpu_budget
from cpu_budget import (
    _cgroup_cpu_quota,
    _probe_cpu_count,
    detect_cpu_count,
    usable_cpu_count,
)


@pytest.fixture
def fake_cgroup(monkeypatch):
    files = {}

    def fake_read(path):
        return files.get(path)

    monkeypatch.setattr(cpu_budget, '_read_text', fake_read)
    return files


def test_cgroup_v2_quota_resolves_to_whole_cpus(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V2_CPU_MAX] = '400000 100000'
    assert _cgroup_cpu_quota() == 4


def test_cgroup_v2_fractional_quota_rounds_up_so_it_never_reaches_zero(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V2_CPU_MAX] = '50000 100000'
    assert _cgroup_cpu_quota() == 1


def test_cgroup_v2_unlimited_quota_reports_no_quota(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V2_CPU_MAX] = 'max 100000'
    assert _cgroup_cpu_quota() is None


def test_cgroup_v2_malformed_quota_reports_no_quota(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V2_CPU_MAX] = 'not-a-number 100000'
    assert _cgroup_cpu_quota() is None


def test_cgroup_v2_present_does_not_fall_through_to_v1(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V2_CPU_MAX] = 'max 100000'
    fake_cgroup[cpu_budget._CGROUP_V1_QUOTA] = '200000'
    fake_cgroup[cpu_budget._CGROUP_V1_PERIOD] = '100000'
    assert _cgroup_cpu_quota() is None


def test_cgroup_v1_quota_and_period_resolve_to_whole_cpus(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V1_QUOTA] = '250000'
    fake_cgroup[cpu_budget._CGROUP_V1_PERIOD] = '100000'
    assert _cgroup_cpu_quota() == 3


def test_cgroup_v1_negative_quota_reports_no_quota(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V1_QUOTA] = '-1'
    fake_cgroup[cpu_budget._CGROUP_V1_PERIOD] = '100000'
    assert _cgroup_cpu_quota() is None


def test_cgroup_v1_malformed_quota_reports_no_quota(fake_cgroup):
    fake_cgroup[cpu_budget._CGROUP_V1_QUOTA] = 'garbage'
    fake_cgroup[cpu_budget._CGROUP_V1_PERIOD] = '100000'
    assert _cgroup_cpu_quota() is None


def test_missing_cgroup_files_report_no_quota(fake_cgroup):
    assert _cgroup_cpu_quota() is None


def test_probe_takes_the_smallest_of_quota_affinity_and_host(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_cgroup_cpu_quota', lambda: 8)
    monkeypatch.setattr(cpu_budget, '_affinity_cpu_count', lambda: 3)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert _probe_cpu_count() == 3


def test_probe_lets_a_container_quota_win_over_the_host_count(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_cgroup_cpu_quota', lambda: 2)
    monkeypatch.setattr(cpu_budget, '_affinity_cpu_count', lambda: 12)
    monkeypatch.setattr(os, 'cpu_count', lambda: 12)
    assert _probe_cpu_count() == 2


def test_probe_falls_back_to_the_host_count_alone(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_cgroup_cpu_quota', lambda: None)
    monkeypatch.setattr(cpu_budget, '_affinity_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 12)
    assert _probe_cpu_count() == 12


def test_probe_ignores_a_reader_that_raises_and_uses_the_rest(monkeypatch):
    def boom():
        raise OSError('cgroup exploded')

    monkeypatch.setattr(cpu_budget, '_cgroup_cpu_quota', boom)
    monkeypatch.setattr(cpu_budget, '_affinity_cpu_count', lambda: 5)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert _probe_cpu_count() == 5


def test_probe_with_nothing_readable_reports_nothing(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_cgroup_cpu_quota', lambda: None)
    monkeypatch.setattr(cpu_budget, '_affinity_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    assert _probe_cpu_count() is None


def test_detect_uses_the_probe_when_it_is_below_the_host(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 4)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (4, 'detected')


def test_detect_reports_host_when_the_probe_matches_the_fallback(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 12)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (12, 'host')


def test_detect_returns_fallback_when_the_probe_raises(monkeypatch):
    def boom():
        raise RuntimeError('probe exploded')

    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', boom)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (12, 'fallback (probe failed)')


def test_detect_raises_a_below_minimum_probe_up_to_the_minimum(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 1)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (2, 'detected (raised to minimum)')


def test_detect_keeps_a_below_minimum_probe_when_the_minimum_is_one(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 1)
    value, source = detect_cpu_count(12, 1)
    assert (value, source) == (1, 'detected')


def test_detect_returns_fallback_when_the_probe_is_zero_or_negative(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 0)
    assert detect_cpu_count(12, 2) == (12, 'fallback (unusable value)')
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: -4)
    assert detect_cpu_count(12, 2) == (12, 'fallback (unusable value)')


def test_detect_returns_fallback_when_the_probe_reports_nothing(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: None)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (12, 'fallback (unusable value)')


def test_detect_returns_fallback_when_the_probe_is_not_an_integer(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 3.5)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (12, 'fallback (unusable value)')


def test_detect_accepts_a_probe_exactly_at_the_minimum(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 2)
    value, source = detect_cpu_count(12, 2)
    assert (value, source) == (2, 'detected')


def test_detect_never_logs_above_info(monkeypatch, caplog):
    def boom():
        raise RuntimeError('probe exploded')

    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', boom)
    with caplog.at_level(logging.DEBUG, logger='cpu_budget'):
        detect_cpu_count(12, 2)
    assert caplog.records
    assert max(record.levelno for record in caplog.records) <= logging.INFO


def test_worker_cap_formula_is_unchanged_when_no_container_limit_is_found(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 12)
    default_cpus, _ = detect_cpu_count(12, 2)
    high_cpus, _ = detect_cpu_count(12, 1)
    assert max(2, default_cpus // 2) == max(2, 12 // 2)
    assert max(1, high_cpus // 3) == max(1, 12 // 3)


def test_usable_cpu_count_matches_the_probe_exactly_with_no_division(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 2)
    monkeypatch.setattr(cpu_budget, 'host_cpu_count', lambda: 12)
    assert usable_cpu_count() == 2


def test_usable_cpu_count_leaves_onnx_alone_when_nothing_restricts_the_host(monkeypatch):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 12)
    monkeypatch.setattr(cpu_budget, 'host_cpu_count', lambda: 12)
    assert usable_cpu_count() is None


def test_usable_cpu_count_reports_nothing_when_the_probe_raises(monkeypatch):
    def boom():
        raise RuntimeError('probe exploded')

    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', boom)
    assert usable_cpu_count() is None


@pytest.mark.parametrize('bad', [None, 0, -1, 3.5])
def test_usable_cpu_count_reports_nothing_for_an_unusable_probe(monkeypatch, bad):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: bad)
    assert usable_cpu_count() is None


def test_usable_cpu_count_never_logs_above_info(monkeypatch, caplog):
    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: None)
    with caplog.at_level(logging.DEBUG, logger='cpu_budget'):
        usable_cpu_count()
    assert caplog.records
    assert max(record.levelno for record in caplog.records) <= logging.INFO


def test_analysis_session_options_pin_intra_op_to_the_usable_cpu_count(monkeypatch):
    from tasks.analysis import song

    monkeypatch.setattr(song, 'usable_cpu_count', lambda: 2)
    assert song._default_sess_options().intra_op_num_threads == 2


def test_analysis_session_options_leave_intra_op_alone_when_nothing_was_found(monkeypatch):
    from tasks.analysis import song

    monkeypatch.setattr(song, 'usable_cpu_count', lambda: None)
    assert song._default_sess_options().intra_op_num_threads == 0


def test_clap_session_options_pin_intra_op_to_the_usable_cpu_count(monkeypatch):
    from tasks import clap_analyzer

    monkeypatch.setattr(clap_analyzer.config, 'CLAP_PYTHON_MULTITHREADS', False)
    monkeypatch.setattr(clap_analyzer, 'usable_cpu_count', lambda: 2)
    assert clap_analyzer._clap_session_options('Audio').intra_op_num_threads == 2


def test_clap_session_options_leave_intra_op_alone_when_nothing_was_found(monkeypatch):
    from tasks import clap_analyzer

    monkeypatch.setattr(clap_analyzer.config, 'CLAP_PYTHON_MULTITHREADS', False)
    monkeypatch.setattr(clap_analyzer, 'usable_cpu_count', lambda: None)
    assert clap_analyzer._clap_session_options('Audio').intra_op_num_threads == 0


def test_clap_python_multithreads_still_pins_onnx_to_one_thread(monkeypatch):
    from tasks import clap_analyzer

    monkeypatch.setattr(clap_analyzer.config, 'CLAP_PYTHON_MULTITHREADS', True)
    monkeypatch.setattr(clap_analyzer, 'usable_cpu_count', lambda: 8)
    assert clap_analyzer._clap_session_options('Audio').intra_op_num_threads == 1


_ENV_KEYS = (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'GOMP_SPINCOUNT',
    'OMP_WAIT_POLICY',
    'AUDIOMUSE_ROLE',
)


def _env_snapshot():
    return {key: os.environ.get(key) for key in _ENV_KEYS}


def _env_restore(snapshot):
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_worker_default_cap_uses_the_container_count(monkeypatch):
    from taskqueue.worker import _apply_thread_caps

    snapshot = _env_snapshot()
    try:
        monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 2)
        monkeypatch.setattr(os, 'cpu_count', lambda: 16)
        cap = _apply_thread_caps('default')
        assert cap == 2
        assert os.environ['OMP_NUM_THREADS'] == '2'
    finally:
        _env_restore(snapshot)


def test_worker_high_cap_uses_the_container_count(monkeypatch):
    from taskqueue.worker import _apply_thread_caps

    snapshot = _env_snapshot()
    try:
        monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 2)
        monkeypatch.setattr(os, 'cpu_count', lambda: 16)
        cap = _apply_thread_caps('high')
        assert cap == 1
        assert os.environ['OMP_NUM_THREADS'] == '1'
    finally:
        _env_restore(snapshot)


def test_worker_cap_keeps_the_host_formula_when_unrestricted(monkeypatch):
    from taskqueue.worker import _apply_thread_caps

    snapshot = _env_snapshot()
    try:
        monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: None)
        monkeypatch.setattr(os, 'cpu_count', lambda: 16)
        cap = _apply_thread_caps('default')
        assert cap == 8
        assert os.environ['OMP_NUM_THREADS'] == '8'
    finally:
        _env_restore(snapshot)


def test_worker_cap_keeps_the_bare_minimum(monkeypatch):
    from taskqueue.worker import _apply_thread_caps

    snapshot = _env_snapshot()
    try:
        monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 1)
        monkeypatch.setattr(os, 'cpu_count', lambda: 16)
        assert _apply_thread_caps('default') == 2
        assert _apply_thread_caps('high') == 1
    finally:
        _env_restore(snapshot)


def test_get_lyrics_threads_uses_the_container_count(monkeypatch):
    from lyrics.lyrics_transcriber import get_lyrics_threads

    monkeypatch.setattr('lyrics.lyrics_transcriber.usable_cpu_count', lambda: 2)
    assert get_lyrics_threads() == 2


def test_get_lyrics_threads_matches_host_when_unrestricted(monkeypatch):
    from lyrics.lyrics_transcriber import get_lyrics_threads

    monkeypatch.setattr('lyrics.lyrics_transcriber.usable_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert get_lyrics_threads() == 8


def test_whisper_resolve_threads_uses_the_container_count(monkeypatch):
    from lyrics import whisper_onnx

    monkeypatch.delenv('LYRICS_WHISPER_INTRA_OP_THREADS', raising=False)
    monkeypatch.setattr(whisper_onnx, 'usable_cpu_count', lambda: 2)
    assert whisper_onnx._resolve_whisper_threads() == 1


def test_whisper_resolve_threads_matches_host_when_unrestricted(monkeypatch):
    from lyrics import whisper_onnx

    monkeypatch.delenv('LYRICS_WHISPER_INTRA_OP_THREADS', raising=False)
    monkeypatch.setattr(whisper_onnx, 'usable_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert whisper_onnx._resolve_whisper_threads() == 5


def test_ivf_manager_max_worker_threads_uses_the_container_count(monkeypatch):
    import tasks.ivf_manager as ivf

    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: 2)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    reloaded = importlib.reload(ivf)
    assert reloaded.MAX_WORKER_THREADS == 1


def test_ivf_manager_max_worker_threads_matches_host_when_unrestricted(monkeypatch):
    import tasks.ivf_manager as ivf

    monkeypatch.setattr(cpu_budget, '_probe_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    reloaded = importlib.reload(ivf)
    assert reloaded.MAX_WORKER_THREADS == 15


def test_artist_gmm_worker_count_uses_the_container_count(monkeypatch):
    from tasks.artist_gmm_manager import _gmm_worker_count

    monkeypatch.setattr('tasks.artist_gmm_manager.INDEX_BUILD_WORKERS', 0)
    monkeypatch.setattr('tasks.artist_gmm_manager.usable_cpu_count', lambda: 2)
    assert _gmm_worker_count(10000) == 1


def test_artist_gmm_worker_count_matches_host_when_unrestricted(monkeypatch):
    from tasks.artist_gmm_manager import _gmm_worker_count

    monkeypatch.setattr('tasks.artist_gmm_manager.INDEX_BUILD_WORKERS', 0)
    monkeypatch.setattr('tasks.artist_gmm_manager.usable_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert _gmm_worker_count(10000) == 8


def test_paged_ivf_query_worker_count_uses_the_container_count(monkeypatch):
    from tasks.paged_ivf import _query_worker_count

    monkeypatch.setattr('tasks.paged_ivf.usable_cpu_count', lambda: 2)
    assert _query_worker_count() == 1


def test_paged_ivf_query_worker_count_matches_host_when_unrestricted(monkeypatch):
    from tasks.paged_ivf import _query_worker_count

    monkeypatch.setattr('tasks.paged_ivf.usable_cpu_count', lambda: None)
    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    assert _query_worker_count() == 8


def test_importing_cpu_budget_pulls_in_no_numeric_library():
    code = (
        'import sys; import cpu_budget; '
        "heavy = [m for m in ('numpy', 'scipy', 'sklearn', 'librosa', 'onnxruntime') "
        'if m in sys.modules]; print(",".join(heavy))'
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, cwd=root, check=True
    )
    assert result.stdout.strip() == ''
