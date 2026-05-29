# tests/unit/test_index_rebuild_integration.py
"""
Integration-level tests for the index rebuild/load glue that the helper-level
unit tests don't reach:

1. ``tasks.artist_gmm_manager.load_artist_index_for_querying`` — the loader where
   the ``db_conn`` NameError lived. Covers the new-BYTEA path, the legacy-JSON
   fallback, and the both-empty reset. A test that calls the loader with a mocked
   DB would have caught that NameError instantly (the bug only manifested when the
   code reached Step B and tried to use an undefined name).

2. ``tasks.analysis._run_all_index_builds`` — the single rebuild orchestrator now
   shared by analysis, cleaning, and collection_manager. Covers: every step runs,
   ``log_fn=None`` is safe, a non-fatal builder failure doesn't abort the rest, and
   a fatal failure (the Voyager audio step) propagates.

These are mock-based — no real Postgres or voyager graph needed. The actual
end-to-end build (``build_and_store_artist_index`` fitting GMMs from DB embeddings
and serializing a real voyager file) is deliberately left to a real-DB integration
test in the docker test stack: driving it through unit mocks would be a brittle
maze (faking voyager.save + open() + three separate SELECTs) with low marginal
value over the codec/storage tests in test_artist_metadata_codec.py.
"""

import json
import sys
import types

import pytest
from unittest.mock import MagicMock, patch, DEFAULT


# artist_gmm_manager imports voyager + sklearn at module load; skip cleanly if absent.
agm = pytest.importorskip("tasks.artist_gmm_manager")
ibh = pytest.importorskip("tasks.index_build_helpers")


@pytest.fixture(autouse=True)
def _reset_artist_globals():
    """Reset the module-level cache before and after each test so state never
    leaks between tests (or from other test files that loaded an index)."""
    def _clear():
        agm.artist_index = None
        agm.artist_map = None
        agm.reverse_artist_map = None
        agm.artist_gmm_params = None
    _clear()
    yield
    _clear()


def _fake_app_helper(conn):
    """A stand-in ``app_helper`` module whose get_db returns our mock connection.

    The loader does ``from app_helper import get_db`` at call time, so injecting
    this into sys.modules intercepts the connection without importing the real
    (heavy) app_helper.
    """
    mod = types.ModuleType("app_helper")
    mod.get_db = MagicMock(return_value=conn)
    return mod


def _conn_returning(row):
    """Mock connection whose single-row SELECT returns ``row`` from fetchone."""
    cur = MagicMock()
    cur.fetchone.return_value = row
    cur.fetchall.return_value = []
    cur.close = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


class TestLoadArtistIndexForQuerying:
    """The loader where the db_conn NameError lived."""

    def test_new_bytea_path_sets_globals_and_passes_real_conn(self):
        """New path: artist_metadata_data blob present -> globals populated from it.

        Critically asserts load_segmented_blob is called with the loader's OWN
        connection object. The historical bug passed ``db_conn`` (undefined in this
        function) -> NameError raised at arg-eval time -> load_segmented_blob never
        called -> this assertion fails. This is the regression guard.
        """
        index_bytes = b"voyager-index-bytes"
        # single-row artist_index_data row: (index_data, legacy_map_json, legacy_gmm_json)
        # legacy JSON columns are '' on the new write path.
        conn, cur = _conn_returning((index_bytes, '', ''))

        fake_map = {0: "Artist A", 1: "Artist B"}
        fake_gmm = {
            "Artist A": {"means": [[0.1, 0.2]], "weights": [1.0], "n_components": 1,
                          "n_features": 2, "n_tracks": 3, "is_few_songs": True, "tracks_hash": "h1"},
            "Artist B": {"means": [[0.3, 0.4]], "weights": [1.0], "n_components": 1,
                          "n_features": 2, "n_tracks": 7, "is_few_songs": False, "tracks_hash": "h2"},
        }
        fake_index = MagicMock()
        fake_index.num_elements = 2

        with patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}), \
             patch.object(ibh, "load_segmented_blob", return_value=b"meta-blob") as mock_load, \
             patch.object(ibh, "unpack_artist_metadata", return_value=(fake_map, fake_gmm)), \
             patch.object(agm, "voyager") as mock_voyager:
            mock_voyager.Index.load.return_value = fake_index
            agm.load_artist_index_for_querying(force_reload=True)

        # The NameError guard: loader must call the metadata loader with ITS conn.
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] is conn, \
            "load_segmented_blob must receive the loader's own connection (regression: db_conn NameError)"

        assert agm.artist_index is fake_index
        assert agm.artist_map == fake_map
        assert agm.artist_gmm_params == fake_gmm
        assert agm.reverse_artist_map == {"Artist A": 0, "Artist B": 1}

    def test_legacy_json_fallback_when_metadata_table_empty(self):
        """Legacy deployment: artist_metadata_data empty -> read legacy JSON columns."""
        index_bytes = b"voyager-index-bytes"
        legacy_map_json = json.dumps({"0": "Artist A", "1": "Artist B"})
        legacy_gmm_json = json.dumps({
            "Artist A": {"means": [[0.1, 0.2]], "weights": [1.0], "n_components": 1, "n_features": 2},
            "Artist B": {"means": [[0.3, 0.4]], "weights": [1.0], "n_components": 1, "n_features": 2},
        })
        conn, cur = _conn_returning((index_bytes, legacy_map_json, legacy_gmm_json))

        fake_index = MagicMock()
        fake_index.num_elements = 2

        with patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}), \
             patch.object(ibh, "load_segmented_blob", return_value=None) as mock_load, \
             patch.object(agm, "voyager") as mock_voyager:
            mock_voyager.Index.load.return_value = fake_index
            agm.load_artist_index_for_querying(force_reload=True)

        mock_load.assert_called_once()  # new path attempted first
        assert agm.artist_index is fake_index
        assert agm.artist_map == {0: "Artist A", 1: "Artist B"}
        assert agm.artist_gmm_params["Artist A"]["n_components"] == 1

    def test_both_sources_empty_resets_cache(self):
        """No metadata anywhere (new table empty + legacy columns '') -> clean reset, no crash."""
        index_bytes = b"voyager-index-bytes"
        conn, cur = _conn_returning((index_bytes, '', ''))

        with patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}), \
             patch.object(ibh, "load_segmented_blob", return_value=None), \
             patch.object(agm, "voyager") as mock_voyager:
            # voyager.Index.load should never be reached, but stub it defensively.
            mock_voyager.Index.load.return_value = MagicMock(num_elements=0)
            agm.load_artist_index_for_querying(force_reload=True)

        assert agm.artist_index is None
        assert agm.artist_map is None
        assert agm.artist_gmm_params is None
        assert agm.reverse_artist_map is None

    def test_num_elements_mismatch_resets_cache(self):
        """If the Voyager element count disagrees with the metadata, abort (don't publish a corrupt index)."""
        index_bytes = b"voyager-index-bytes"
        conn, cur = _conn_returning((index_bytes, '', ''))
        fake_map = {0: "Artist A", 1: "Artist B"}          # 2 artists
        fake_gmm = {"Artist A": {"means": [[0.1]], "weights": [1.0], "n_components": 1, "n_features": 1},
                    "Artist B": {"means": [[0.2]], "weights": [1.0], "n_components": 1, "n_features": 1}}
        fake_index = MagicMock()
        fake_index.num_elements = 5                          # mismatch: 5 != 2

        with patch.dict(sys.modules, {"app_helper": _fake_app_helper(conn)}), \
             patch.object(ibh, "load_segmented_blob", return_value=b"meta-blob"), \
             patch.object(ibh, "unpack_artist_metadata", return_value=(fake_map, fake_gmm)), \
             patch.object(agm, "voyager") as mock_voyager:
            mock_voyager.Index.load.return_value = fake_index
            agm.load_artist_index_for_querying(force_reload=True)

        assert agm.artist_index is None
        assert agm.artist_map is None


# ---------------------------------------------------------------------------
# Orchestrator: tasks.analysis._run_all_index_builds
# ---------------------------------------------------------------------------

analysis_mod = None
try:
    import tasks.analysis as analysis_mod  # noqa: E402  (heavy: librosa/onnx)
except Exception:
    analysis_mod = None


_BUILDER_NAMES = [
    "build_and_store_voyager_index",
    "build_and_store_clap_index",
    "build_and_store_lyrics_index",
    "build_and_store_lyrics_axes_index",
    "build_and_store_sem_grove_index",
    "build_and_store_artist_index",
    "build_and_store_map_projection",
    "build_and_store_artist_projection",
]


@pytest.mark.skipif(analysis_mod is None, reason="tasks.analysis (librosa/onnx) unavailable in this env")
class TestRunAllIndexBuilds:
    """The single rebuild entry point shared by analysis, cleaning, collection_manager."""

    def _patched(self):
        """patch.multiple over the orchestrator's module-level deps + all 8 builders."""
        targets = {name: DEFAULT for name in _BUILDER_NAMES}
        targets["get_db"] = DEFAULT
        targets["redis_conn"] = DEFAULT
        targets["_release_freed_ram_to_os"] = DEFAULT
        return patch.multiple(analysis_mod, **targets)

    def test_all_eight_builders_run_with_log_fn_none(self):
        with self._patched() as mocks:
            analysis_mod._run_all_index_builds(log_fn=None)
        for name in _BUILDER_NAMES:
            assert mocks[name].called, f"{name} was not invoked by the orchestrator"
        # reload published + RAM released at the end
        assert mocks["redis_conn"].publish.called
        assert mocks["_release_freed_ram_to_os"].called

    def test_non_fatal_failure_does_not_abort_remaining_builders(self):
        with self._patched() as mocks:
            mocks["build_and_store_clap_index"].side_effect = RuntimeError("clap boom")
            # must NOT raise — CLAP is non-fatal
            analysis_mod._run_all_index_builds(log_fn=None)
            # everything after CLAP still ran
            assert mocks["build_and_store_lyrics_index"].called
            assert mocks["build_and_store_sem_grove_index"].called
            assert mocks["build_and_store_artist_index"].called
            assert mocks["build_and_store_artist_projection"].called

    def test_fatal_voyager_failure_propagates_and_aborts(self):
        with self._patched() as mocks:
            mocks["build_and_store_voyager_index"].side_effect = RuntimeError("fatal voyager")
            with pytest.raises(RuntimeError, match="fatal voyager"):
                analysis_mod._run_all_index_builds(log_fn=None)
            # aborted at the first (fatal) step — later builders never ran
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
        assert 95 in progresses          # initial "Performing final index rebuild..."
        assert any("CLAP" in m for m in messages)
        assert any("artist similarity" in m.lower() for m in messages)
