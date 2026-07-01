# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Artist-index loading and the _run_all_index_builds orchestrator.

Covers load_artist_index_for_querying wiring the artist globals from the paged
IVF index and metadata, and the orchestrator that runs all index builders.

Main Features:
* Artist index load sets or resets the module globals depending on IVF presence
  and metadata availability
* The orchestrator invokes all eight builders and publishes progress
* A non-fatal builder failure continues; a fatal IVF failure propagates and aborts
"""

import sys
import types
from contextlib import ExitStack, contextmanager

import pytest
from unittest.mock import MagicMock, patch


agm = pytest.importorskip("tasks.artist_gmm_manager")
ibh = pytest.importorskip("tasks.index_build_helpers")


@pytest.fixture(autouse=True)
def _reset_artist_globals():
    def _clear():
        agm.artist_index = None
        agm.artist_map = None
        agm.reverse_artist_map = None
        agm.artist_gmm_params = None

    _clear()
    yield
    _clear()


def _fake_app_helper(conn):
    mod = types.ModuleType("app_helper")
    mod.get_db = MagicMock(return_value=conn)
    return mod


def _conn_returning(row):
    cur = MagicMock()
    cur.fetchone.return_value = row
    cur.fetchall.return_value = []
    cur.close = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


class TestLoadArtistIndexForQuerying:
    def test_ivf_path_sets_globals(self):
        conn, cur = _conn_returning(None)
        fake_map = {0: "Artist A", 1: "Artist B"}
        fake_gmm = {
            "Artist A": {
                "means": [[0.1, 0.2]],
                "weights": [1.0],
                "n_components": 1,
                "n_features": 2,
                "n_tracks": 3,
                "is_few_songs": True,
                "tracks_hash": "h1",
            },
            "Artist B": {
                "means": [[0.3, 0.4]],
                "weights": [1.0],
                "n_components": 1,
                "n_features": 2,
                "n_tracks": 7,
                "is_few_songs": False,
                "tracks_hash": "h2",
            },
        }
        fake_index = MagicMock()
        fake_index.__len__.return_value = len(fake_map)
        with (
            patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}),
            patch("tasks.paged_ivf.has_paged_ivf", return_value=True),
            patch("tasks.paged_ivf.load_paged_ivf_index", return_value=(fake_index, fake_map, {})),
            patch.object(ibh, "load_segmented_blob", return_value=b"meta-blob"),
            patch.object(ibh, "unpack_artist_metadata", return_value=(fake_map, fake_gmm)),
        ):
            agm.load_artist_index_for_querying(force_reload=True)

        assert agm.artist_index is fake_index
        assert agm.artist_map == fake_map
        assert agm.artist_gmm_params == fake_gmm
        assert agm.reverse_artist_map == {"Artist A": 0, "Artist B": 1}

    def test_no_ivf_index_resets_cache(self):
        conn, cur = _conn_returning(None)
        with (
            patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}),
            patch("tasks.paged_ivf.has_paged_ivf", return_value=False),
        ):
            agm.load_artist_index_for_querying(force_reload=True)

        assert agm.artist_index is None
        assert agm.artist_map is None
        assert agm.artist_gmm_params is None
        assert agm.reverse_artist_map is None

    def test_missing_metadata_resets_cache(self):
        conn, cur = _conn_returning(None)
        fake_index = MagicMock()
        with (
            patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}),
            patch("tasks.paged_ivf.has_paged_ivf", return_value=True),
            patch("tasks.paged_ivf.load_paged_ivf_index", return_value=(fake_index, {0: "A"}, {})),
            patch.object(ibh, "load_segmented_blob", return_value=None),
        ):
            agm.load_artist_index_for_querying(force_reload=True)

        assert agm.artist_index is None
        assert agm.artist_map is None


analysis_mod = None
try:
    import tasks.analysis as analysis_mod  # noqa: E402  (heavy: librosa/onnx)
    import tasks.ivf_manager  # noqa: F401  (builder modules patched in _patched)
    import tasks.clap_text_search  # noqa: F401
    import tasks.lyrics_manager  # noqa: F401
    import tasks.sem_grove_manager  # noqa: F401
except Exception:
    analysis_mod = None


_BUILDER_NAMES = [
    "build_and_store_ivf_index",
    "build_and_store_clap_index",
    "build_and_store_lyrics_index",
    "build_and_store_lyrics_axes_index",
    "build_and_store_sem_grove_index",
    "build_and_store_artist_index",
    "build_and_store_map_projection",
    "build_and_store_artist_projection",
]

_BUILDER_SOURCE_MODULES = {
    "build_and_store_ivf_index": "tasks.ivf_manager",
    "build_and_store_clap_index": "tasks.clap_text_search",
    "build_and_store_lyrics_index": "tasks.lyrics_manager",
    "build_and_store_lyrics_axes_index": "tasks.lyrics_manager",
    "build_and_store_sem_grove_index": "tasks.sem_grove_manager",
    "build_and_store_artist_index": "tasks.artist_gmm_manager",
    "build_and_store_map_projection": "tasks.analysis",
    "build_and_store_artist_projection": "tasks.analysis",
}


@pytest.mark.skipif(
    analysis_mod is None, reason="tasks.analysis (librosa/onnx) unavailable in this env"
)
class TestRunAllIndexBuilds:
    @contextmanager
    def _patched(self):
        with ExitStack() as stack:
            mocks = {}
            for name, module in _BUILDER_SOURCE_MODULES.items():
                mocks[name] = stack.enter_context(patch(f"{module}.{name}"))
            for name in ("get_db", "redis_conn", "_release_freed_ram_to_os"):
                mocks[name] = stack.enter_context(patch.object(analysis_mod, name))
            yield mocks

    def test_all_eight_builders_run_with_log_fn_none(self):
        with self._patched() as mocks:
            analysis_mod._run_all_index_builds(log_fn=None)
        for name in _BUILDER_NAMES:
            assert mocks[name].called, f"{name} was not invoked by the orchestrator"
        assert mocks["redis_conn"].publish.called
        assert mocks["_release_freed_ram_to_os"].called

    def test_non_fatal_failure_does_not_abort_remaining_builders(self):
        with self._patched() as mocks:
            mocks["build_and_store_clap_index"].side_effect = RuntimeError("clap boom")
            analysis_mod._run_all_index_builds(log_fn=None)
            assert mocks["build_and_store_lyrics_index"].called
            assert mocks["build_and_store_sem_grove_index"].called
            assert mocks["build_and_store_artist_index"].called
            assert mocks["build_and_store_artist_projection"].called

    def test_fatal_ivf_failure_propagates_and_aborts(self):
        with self._patched() as mocks:
            mocks["build_and_store_ivf_index"].side_effect = RuntimeError("fatal ivf")
            with pytest.raises(RuntimeError, match="fatal ivf"):
                analysis_mod._run_all_index_builds(log_fn=None)
            assert not mocks["build_and_store_clap_index"].called
            assert not mocks["build_and_store_artist_index"].called

    def test_log_fn_receives_progress_banners(self):
        calls = []

        def log_fn(message, progress):
            calls.append((message, progress))

        with self._patched():
            analysis_mod._run_all_index_builds(log_fn=log_fn)

        progresses = [p for _, p in calls]
        messages = [m for m, _ in calls]
        assert 95 in progresses
        assert any("CLAP" in m for m in messages)
        assert any("artist similarity" in m.lower() for m in messages)
