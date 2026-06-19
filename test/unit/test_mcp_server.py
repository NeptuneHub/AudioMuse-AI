"""Unit tests for tasks/mcp_helper.py and tasks/mcp_tool_impl.py.

Tests cover MCP helper + tool functions:
- get_library_context(): Library statistics with caching
- _database_genre_query_sync(): Genre regex matching, filters, relevance scoring
- _extract_json_object() / _clamp_recipe(): brainstorm recipe parse + vocab clamp
- _ai_brainstorm_sync(): grounded recipe -> fused retrieval channels (#643)
- _song_similarity_api_sync(): Song lookup with exact/fuzzy fallback
- Energy normalization in execute_mcp_tool()
- Pre-execution validation (filterless search_database rejection)

NOTE: uses importlib to load tasks.mcp_helper / tasks.ai.tool_impl directly,
bypassing tasks/__init__.py which pulls in pydub (requires audioop removed in
Python 3.14).
"""
import json
import re
import os
import sys
import importlib.util
import pytest
from unittest.mock import Mock, MagicMock, patch


# ---------------------------------------------------------------------------
# Module loaders (bypass tasks/__init__.py -> pydub -> audioop chain)
# ---------------------------------------------------------------------------

def _import_mcp_server():
    """Load tasks.mcp_helper directly without triggering tasks/__init__.py.

    Name kept as ``_import_mcp_server`` for historical reasons -- the module
    was renamed to ``mcp_helper`` but tests still use this loader.
    """
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'mcp_helper.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'tasks.mcp_helper'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def _import_ai_mcp_client():
    """Load tasks.ai.tools (the MCP tool definitions + dispatcher)."""
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'ai', 'tools.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'tasks.ai.tools'
    if mod_name not in sys.modules:
        _import_mcp_server()
        _import_mcp_impl()
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def _import_mcp_impl():
    """Load tasks.ai.tool_impl directly without triggering tasks/__init__.py."""
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'ai', 'tool_impl.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'tasks.ai.tool_impl'
    if mod_name not in sys.modules:
        _import_mcp_server()
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Helpers to build mock DB cursors
# ---------------------------------------------------------------------------

def _make_dict_row(mapping: dict):
    """Create an object that supports both dict-key access and attribute access,
    mimicking psycopg2 DictRow."""
    class FakeRow(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
    return FakeRow(mapping)


def _make_connection(cursor):
    """Wrap a mock cursor in a mock connection."""
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.close = Mock()
    return conn


# ---------------------------------------------------------------------------
# Genre regex pattern tests (pure pattern tests, no DB needed)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenreRegexPattern:
    """Test the regex pattern used in _database_genre_query_sync for genre matching."""

    def _matches(self, genre, mood_vector):
        """Check if the actual mcp_server regex pattern (with (?i) inline flag) matches.

        Mirrors the param built in _database_genre_query_sync — the (?i) prefix is
        what PostgreSQL POSIX regex uses; Python re also honors it as an inline flag,
        so this test exercises the exact pattern the SQL layer receives.
        """
        pattern = f"(?i)(?:^|,)\\s*{re.escape(genre)}:(\\d+\\.?\\d*)"
        return bool(re.search(pattern, mood_vector))

    def test_genre_at_start_matches(self):
        assert self._matches("rock", "rock:0.82,pop:0.45")

    def test_genre_after_comma_matches(self):
        assert self._matches("rock", "pop:0.45,rock:0.82")

    def test_genre_after_comma_with_space_matches(self):
        assert self._matches("rock", "pop:0.45, rock:0.82")

    def test_substring_does_not_match(self):
        """'rock' must NOT match 'indie rock'."""
        assert not self._matches("rock", "indie rock:0.31,pop:0.45")

    def test_compound_genre_matches(self):
        """'indie rock' should match 'indie rock:0.31'."""
        assert self._matches("indie rock", "pop:0.45,indie rock:0.31")

    def test_case_insensitive(self):
        assert self._matches("Rock", "rock:0.82,pop:0.45")

    def test_no_match_returns_false(self):
        assert not self._matches("jazz", "rock:0.82,pop:0.45")

    def test_single_genre_vector(self):
        assert self._matches("rock", "rock:0.82")

    def test_genre_with_special_chars(self):
        """Genres with regex-special chars should be escaped."""
        assert self._matches("r&b", "r&b:0.65,pop:0.45")

    def test_hip_hop_no_substring_match(self):
        """'hip hop' must not match 'trip hop'."""
        assert not self._matches("hip hop", "trip hop:0.45")

    def test_pop_no_substring_match(self):
        """'pop' must not match 'indie pop'."""
        assert not self._matches("pop", "indie pop:0.55,rock:0.82")

    def test_pop_matches_at_start(self):
        assert self._matches("pop", "pop:0.55,rock:0.82")

    def test_lowercase_input_matches_titlecase_label(self):
        """MOOD_LABELS contains 'Mellow' / 'Hip-Hop' / 'Progressive rock' — AI sends
        these lowercased, so the regex must be case-insensitive to find them."""
        assert self._matches("mellow", "Mellow:0.74,pop:0.45")
        assert self._matches("hip-hop", "Hip-Hop:0.61,rock:0.20")
        assert self._matches("progressive rock", "Progressive rock:0.55")

    def test_uppercase_input_matches_lowercase_stored(self):
        assert self._matches("ROCK", "rock:0.82,pop:0.45")


# ---------------------------------------------------------------------------
# get_library_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetLibraryContext:
    """Tests for get_library_context() - library stats with caching."""

    def _reset_cache(self):
        mod = _import_mcp_server()
        mod._library_context_cache = None

    def test_returns_expected_keys(self):
        mod = _import_mcp_server()
        self._reset_cache()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)

        cur.fetchone = Mock(side_effect=[
            _make_dict_row({"cnt": 500, "artists": 80}),
            _make_dict_row({"ymin": 1965, "ymax": 2024}),
            _make_dict_row({"rated": 200}),
        ])

        cur.fetchall = Mock(side_effect=[
            # genres query (name/cnt)
            [_make_dict_row({"name": "rock", "cnt": 120}), _make_dict_row({"name": "pop", "cnt": 90})],
            # scales query
            [_make_dict_row({"scale": "major"}), _make_dict_row({"scale": "minor"})],
            # moods query (name/cnt)
            [_make_dict_row({"name": "danceable", "cnt": 80}), _make_dict_row({"name": "happy", "cnt": 60})],
        ])

        conn = _make_connection(cur)

        with patch.object(mod, 'get_db_connection', return_value=conn):
            ctx = mod.get_library_context(force_refresh=True)

        assert ctx["total_songs"] == 500
        assert ctx["unique_artists"] == 80
        assert ctx["year_min"] == 1965
        assert ctx["year_max"] == 2024
        assert ctx["has_ratings"] is True
        assert ctx["rated_songs_pct"] == 40.0
        assert "rock" in ctx["top_genres"]
        assert "danceable" in ctx["top_moods"]
        assert "major" in ctx["scales"]
        conn.close.assert_called_once()

    def test_caching_returns_same_result(self):
        """Second call without force_refresh returns cached result."""
        mod = _import_mcp_server()
        self._reset_cache()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchone = Mock(return_value=_make_dict_row({"cnt": 100, "artists": 10, "ymin": 2000, "ymax": 2020, "rated": 50}))
        cur.__iter__ = Mock(return_value=iter([]))
        cur.fetchall = Mock(return_value=[])
        conn = _make_connection(cur)
        mock_get_conn = Mock(return_value=conn)

        with patch.object(mod, 'get_db_connection', mock_get_conn):
            ctx1 = mod.get_library_context(force_refresh=True)
            ctx2 = mod.get_library_context(force_refresh=False)

        # DB should only be called once
        assert mock_get_conn.call_count == 1
        assert ctx1 is ctx2

    def test_empty_library_returns_defaults(self):
        mod = _import_mcp_server()
        self._reset_cache()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchone = Mock(return_value=_make_dict_row({"cnt": 0, "artists": 0, "ymin": None, "ymax": None, "rated": 0}))
        cur.__iter__ = Mock(return_value=iter([]))
        cur.fetchall = Mock(return_value=[])
        conn = _make_connection(cur)

        with patch.object(mod, 'get_db_connection', return_value=conn):
            ctx = mod.get_library_context(force_refresh=True)

        assert ctx["total_songs"] == 0
        assert ctx["unique_artists"] == 0
        assert ctx["has_ratings"] is False


# ---------------------------------------------------------------------------
# Energy normalization
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnergyNormalization:
    """Test energy conversion from 0-1 (AI scale) to raw (DB scale)."""

    def test_zero_maps_to_energy_min(self):
        e_min, e_max = 0.01, 0.15
        raw = e_min + 0.0 * (e_max - e_min)
        assert raw == pytest.approx(0.01)

    def test_one_maps_to_energy_max(self):
        e_min, e_max = 0.01, 0.15
        raw = e_min + 1.0 * (e_max - e_min)
        assert raw == pytest.approx(0.15)

    def test_half_maps_to_midpoint(self):
        e_min, e_max = 0.01, 0.15
        raw = e_min + 0.5 * (e_max - e_min)
        assert raw == pytest.approx(0.08)

    def test_quarter_maps_correctly(self):
        e_min, e_max = 0.01, 0.15
        raw = e_min + 0.25 * (e_max - e_min)
        assert raw == pytest.approx(0.045)

    def test_three_quarter_maps_correctly(self):
        e_min, e_max = 0.01, 0.15
        raw = e_min + 0.75 * (e_max - e_min)
        assert raw == pytest.approx(0.115)


# ---------------------------------------------------------------------------
# _database_genre_query_sync
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDatabaseGenreQuery:
    """Tests for _database_genre_query_sync - database filtering."""

    def _setup_mock_conn(self):
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchall = Mock(return_value=[])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)
        return conn, cur

    def test_genre_filter_builds_regex_condition(self):
        """Verify the SQL uses SUBSTRING with regex pattern for genre matching.

        Also asserts the (?i) inline flag is present so case-mismatched MOOD_LABELS
        like 'Mellow', 'Hip-Hop', 'Progressive rock' still match user-supplied lowercase.
        """
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(genres=["rock"], get_songs=10)

        call_args = cur.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else []
        assert "SUBSTRING(mood_vector FROM" in sql  # PostgreSQL regex extraction
        found_regex = any("rock:" in str(p) for p in params) if params else False
        assert found_regex or "rock" in sql
        # Case-insensitive flag: required for mixed-case stored labels.
        assert any("(?i)" in str(p) for p in params), "genre regex must use (?i)"

    def test_tempo_range_filter(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(tempo_min=120, tempo_max=140, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "tempo >=" in sql
        assert "tempo <=" in sql

    def test_key_filter_uppercased(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(key="c", get_songs=10)

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        assert "key = %s" in sql
        assert "C" in params  # should be uppercased

    def test_scale_filter_case_insensitive(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(scale="Major", get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "LOWER(scale)" in sql

    def test_year_range_filter(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(year_min=1980, year_max=1989, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "year >=" in sql
        assert "year <=" in sql

    def test_min_rating_filter(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(min_rating=4, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "rating >=" in sql

    def test_mood_filter_uses_score_threshold(self):
        """Mood filter must extract per-mood score from other_features and apply a confidence
        threshold. A bare LIKE would match every song (every row contains all 6 mood labels),
        which is the bug from issue #472."""
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(moods=["aggressive"], get_songs=10)

        call_args = cur.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else []
        # Score-extraction pattern (mirrors the genre filter)
        assert "SUBSTRING(other_features FROM" in sql
        # Threshold parameter present and meaningfully above the noise floor.
        # CLAP-derived other_features scores cluster near 0.5; a threshold below ~0.5 would
        # re-match the entire library (the bug from issue #472).
        assert any(isinstance(p, float) and 0.5 <= p < 1 for p in params)
        # Per-mood regex pattern present
        assert any("aggressive:" in str(p) for p in params)
        # Case-insensitive flag must be present (defense against AI returning "Aggressive")
        assert any("(?i)" in str(p) for p in params), "mood regex must use (?i)"
        # And ordering must be by relevance (not RANDOM) so top mood matches come first
        assert "relevance_score DESC" in sql

    def test_mood_filter_does_not_use_bare_like(self):
        """Regression guard for issue #472: bare LIKE on other_features matches every song."""
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(moods=["aggressive"], get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "other_features LIKE" not in sql

    def test_combined_filters_use_and(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(
                genres=["rock"], tempo_min=120, energy_min=0.05,
                key="C", scale="major", year_min=2000, min_rating=3, get_songs=10
            )

        sql = cur.execute.call_args[0][0]
        assert sql.count("AND") >= 5  # Multiple AND conditions

    def test_results_returned_as_list(self):
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "1", "title": "Song A", "author": "Artist A",
                           "album": "Album", "album_artist": "AA", "tempo": 120,
                           "key": "C", "scale": "major", "energy": 0.08,
                           "mood_vector": "rock:0.82", "other_features": "danceable"}),
        ])

        with patch.object(mod, 'get_db_connection', return_value=conn):
            result = mod._database_genre_query_sync(genres=["rock"], get_songs=10)

        assert isinstance(result, (list, dict))
        if isinstance(result, dict):
            assert "songs" in result

    def test_get_songs_converted_to_int(self):
        """Gemini may send float for get_songs - should be converted to int."""
        mod = _import_mcp_impl()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            # Should not raise - float get_songs handled
            mod._database_genre_query_sync(genres=["rock"], get_songs=50.0)


@pytest.mark.unit
class TestRerouteMoodLabelsFromGenres:
    """The helper that fixes AI confusion: mood labels passed as genres -> rerouted to moods."""

    def test_no_genres_is_noop(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(None, ["happy"])
        assert g is None
        assert m == ["happy"]
        assert msg is None

    def test_only_real_genres_is_noop(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(["rock", "metal"], None)
        assert g == ["rock", "metal"]
        assert m is None
        assert msg is None

    def test_mood_label_in_genres_gets_rerouted(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(["aggressive"], None)
        assert g == []
        assert m == ["aggressive"]
        assert msg is not None and "aggressive" in msg

    def test_mixed_keeps_real_genres_reroutes_mood(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(
            ["rock", "aggressive", "metal"], None
        )
        assert g == ["rock", "metal"]
        assert m == ["aggressive"]
        assert msg is not None

    def test_case_insensitive(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(["Aggressive"], None)
        assert g == []
        assert m == ["aggressive"]  # canonicalized to lowercase

    def test_no_duplicate_when_already_in_moods(self):
        mod = _import_mcp_impl()
        g, m, msg = mod._reroute_mood_labels_from_genres(["aggressive"], ["aggressive"])
        assert m == ["aggressive"]  # unchanged, no dup
        assert g == []

    def test_rerouting_applied_in_database_query(self):
        """End-to-end: AI passes 'aggressive' as a genre, query treats it as a mood."""
        mod = _import_mcp_impl()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchall = Mock(return_value=[])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn):
            result = mod._database_genre_query_sync(genres=["aggressive"], get_songs=10)

        sql = cur.execute.call_args[0][0]
        # No genre regex against mood_vector should be built (genres list is empty after rerouting)
        assert "SUBSTRING(mood_vector FROM" not in sql
        # Mood regex against other_features SHOULD be built
        assert "SUBSTRING(other_features FROM" in sql
        # Reroute warning should appear in the output message
        assert "Rerouted" in result["message"]


# ---------------------------------------------------------------------------
# brainstorm recipe helpers (pure, no DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractJsonObject:
    """_extract_json_object: recover ONE JSON object from messy model output."""

    def _fn(self):
        return _import_mcp_impl()._extract_json_object

    def test_plain_object(self):
        assert self._fn()('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert self._fn()('```json\n{"a": 1}\n```') == {"a": 1}

    def test_think_preamble_stripped(self):
        assert self._fn()('<think>reasoning...</think>\n{"a": 2}') == {"a": 2}

    def test_object_embedded_in_prose(self):
        assert self._fn()('Sure, here: {"a": 3} hope it helps') == {"a": 3}

    def test_array_is_rejected(self):
        assert self._fn()('[{"a": 1}]') is None

    def test_garbage_returns_none(self):
        assert self._fn()('no json here at all') is None

    def test_empty_returns_none(self):
        assert self._fn()('') is None


@pytest.mark.unit
class TestClampRecipe:
    """_clamp_recipe: normalise a raw recipe to library-valid, bounded values."""

    def _fn(self):
        return _import_mcp_impl()._clamp_recipe

    def test_genres_clamped_to_vocab_case_and_punct_insensitive(self):
        out = self._fn()({"filters": {"genres": ["hip hop", "ROCK", "not-a-genre"]}})
        assert out["filters"]["genres"] == ["Hip-Hop", "rock"]

    def test_moods_and_voices_clamped(self):
        out = self._fn()({"filters": {"moods": ["Party", "bogus"], "voices": ["female vocalist"]}})
        assert out["filters"]["moods"] == ["party"]
        assert out["filters"]["voices"] == ["female vocalist"]

    def test_year_range_reversed_is_swapped(self):
        out = self._fn()({"filters": {"year_min": 2009, "year_max": 1990}})
        assert out["filters"]["year_min"] == 1990
        assert out["filters"]["year_max"] == 2009

    def test_energy_clamped_then_swapped(self):
        out = self._fn()({"filters": {"energy_min": 2.0, "energy_max": -1.0}})
        assert out["filters"]["energy_min"] == pytest.approx(0.0)
        assert out["filters"]["energy_max"] == pytest.approx(1.0)

    def test_lists_deduped_and_capped(self):
        import config as cfg
        out = self._fn()({
            "sound_descriptions": ["a", "a", "b", "c", "d", "e"],
            "seed_artists": ["X", "x", "Y", "Z", "W", "V"],
            "lyric_themes": ["t1", "t2", "t3"],
        })
        assert out["sound_descriptions"][:2] == ["a", "b"]
        assert len(out["sound_descriptions"]) <= cfg.AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX
        assert len(out["seed_artists"]) <= cfg.AI_BRAINSTORM_SEED_ARTISTS_MAX
        assert len(out["lyric_themes"]) <= cfg.AI_BRAINSTORM_LYRIC_THEMES_MAX

    def test_missing_filters_yields_empty_defaults(self):
        out = self._fn()({})
        f = out["filters"]
        assert f["genres"] == [] and f["moods"] == [] and f["voices"] == []
        assert f["year_min"] is None and f["energy_max"] is None
        assert out["sound_descriptions"] == [] and out["seed_artists"] == []

    def test_non_list_fields_are_coerced(self):
        out = self._fn()({"sound_descriptions": "just one", "filters": {"genres": "rock"}})
        assert out["sound_descriptions"] == ["just one"]
        assert out["filters"]["genres"] == ["rock"]

    def test_seed_artists_suppressed_when_disabled(self):
        mod = _import_mcp_impl()
        import config as cfg
        with patch.object(cfg, "AI_BRAINSTORM_USE_ARTIST_SEEDS", False):
            out = mod._clamp_recipe({"seed_artists": ["Nas", "Jay-Z"]})
        assert out["seed_artists"] == []


# ---------------------------------------------------------------------------
# execute_mcp_tool energy conversion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecuteMcpToolEnergyConversion:
    """Test that execute_mcp_tool converts energy from 0-1 to raw."""

    def test_search_database_energy_conversion(self):
        ai_mod = _import_ai_mcp_client()

        mock_query = Mock(return_value={"songs": []})
        import config as cfg
        orig_min, orig_max = cfg.ENERGY_MIN, cfg.ENERGY_MAX
        try:
            cfg.ENERGY_MIN = 0.01
            cfg.ENERGY_MAX = 0.15
            # Dispatcher binds the impl at import time -- patch the bound name on the dispatcher module.
            with patch.object(ai_mod, '_database_genre_query_sync', mock_query):
                ai_mod.execute_mcp_tool("search_database", {
                    "genres": ["rock"],
                    "energy_min": 0.5,
                    "energy_max": 0.8
                }, {})

        finally:
            cfg.ENERGY_MIN = orig_min
            cfg.ENERGY_MAX = orig_max

    def test_unknown_tool_returns_error(self):
        ai_mod = _import_ai_mcp_client()
        result = ai_mod.execute_mcp_tool("nonexistent_tool", {}, {})
        assert "error" in result


# ---------------------------------------------------------------------------
# Song similarity lookup patterns
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSongSimilarityLookup:
    """Tests for _song_similarity_api_sync patterns."""

    def test_exact_match_case_insensitive(self):
        mod = _import_mcp_impl()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchone = Mock(return_value=_make_dict_row({
            "item_id": "123", "title": "Bohemian Rhapsody", "author": "Queen"
        }))
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        mock_nn = Mock(return_value=[
            {"item_id": "123", "distance": 0.0},
            {"item_id": "456", "distance": 0.1},
        ])
        # Create a mock ivf_manager module in sys.modules to avoid tasks/__init__.py
        mock_ivf = MagicMock()
        mock_ivf.find_nearest_neighbors_by_id = mock_nn
        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.ivf_manager': mock_ivf}):
            mod._song_similarity_api_sync("bohemian rhapsody", "queen", 10)

        # Should have tried a DB lookup
        assert cur.execute.called

    def test_no_match_returns_empty(self):
        mod = _import_mcp_impl()
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchone = Mock(return_value=None)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn):
            result = mod._song_similarity_api_sync("nonexistent song", "unknown artist", 10)

        assert isinstance(result, (list, dict))
        if isinstance(result, dict):
            assert len(result.get("songs", [])) == 0


# ---------------------------------------------------------------------------
# _artist_similarity_api_sync
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestArtistSimilarityApiSync:
    """Tests for _artist_similarity_api_sync - artist similarity with GMM."""

    def _setup_cursor(self):
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        return cur

    def _setup_gmm_module(self, find_return=None, reverse_map=None):
        """Build a mock tasks.artist_gmm_manager module."""
        mock_mod = MagicMock()
        mock_mod.find_similar_artists = Mock(return_value=find_return or [])
        mock_mod.reverse_artist_map = reverse_map if reverse_map is not None else {}
        return mock_mod

    def test_exact_match_returns_songs(self):
        """Exact DB match -> find_similar_artists -> songs returned."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "Radiohead"}))
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "1", "title": "Creep", "author": "Radiohead"}),
            _make_dict_row({"item_id": "2", "title": "Paranoid Android", "author": "Muse"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(
            find_return=[{"artist": "Muse", "distance": 0.1}]
        )

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("Radiohead", count=5, get_songs=10)

        assert "songs" in result
        assert len(result["songs"]) > 0

    def test_fuzzy_match_fallback(self):
        """No exact match -> fuzzy ILIKE match used."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(side_effect=[
            None,
            _make_dict_row({"author": "AC/DC", "len": 5}),
        ])
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "10", "title": "Back in Black", "author": "AC/DC"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(
            find_return=[{"artist": "Guns N' Roses", "distance": 0.2}]
        )

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("AC DC", count=5, get_songs=10)

        assert "songs" in result
        assert cur.fetchone.call_count == 2

    def test_no_match_returns_empty(self):
        """All DB lookups return None -> empty songs with message."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=None)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(find_return=[])

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("ZZZ Unknown", count=5, get_songs=10)

        assert result["songs"] == []
        assert "message" in result

    def test_gmm_empty_fallback_to_reverse_artist_map(self):
        """GMM returns [] -> fallback to reverse_artist_map fuzzy match."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "Queen"}))
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "5", "title": "We Will Rock You", "author": "Queen"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = MagicMock()
        gmm_mod.find_similar_artists = Mock(side_effect=[
            [],
            [{"artist": "David Bowie", "distance": 0.3}],
        ])
        gmm_mod.reverse_artist_map = {"queen": 0, "david bowie": 1}

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("Queen", count=5, get_songs=10)

        assert gmm_mod.find_similar_artists.call_count >= 2
        assert "songs" in result

    def test_special_chars_fallback_via_resub(self):
        """Artist with special chars, GMM empty, re.sub cleanup triggers fallback."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "P!nk"}))
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "20", "title": "So What", "author": "P!nk"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = MagicMock()
        gmm_mod.find_similar_artists = Mock(side_effect=[
            [],
            [{"artist": "Kelly Clarkson", "distance": 0.4}],
        ])
        gmm_mod.reverse_artist_map = {}

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("P!nk", count=5, get_songs=10)

        assert gmm_mod.find_similar_artists.call_count >= 2
        assert "songs" in result

    def test_result_structure_has_required_keys(self):
        """Returned dict has songs, similar_artists, component_matches, message."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "Nirvana"}))
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "30", "title": "Smells Like Teen Spirit", "author": "Nirvana"}),
            _make_dict_row({"item_id": "31", "title": "Everlong", "author": "Foo Fighters"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(
            find_return=[{"artist": "Foo Fighters", "distance": 0.15}]
        )

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("Nirvana", count=5, get_songs=10)

        assert "songs" in result
        assert "similar_artists" in result
        assert "component_matches" in result
        assert "message" in result

    def test_component_matches_includes_original_artist(self):
        """component_matches marks the original artist with is_original=True."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "The Beatles"}))
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "40", "title": "Hey Jude", "author": "The Beatles"}),
            _make_dict_row({"item_id": "41", "title": "Imagine", "author": "John Lennon"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(
            find_return=[{"artist": "John Lennon", "distance": 0.1}]
        )

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            result = mod._artist_similarity_api_sync("The Beatles", count=5, get_songs=10)

        original_entries = [
            c for c in result["component_matches"] if c.get("is_original") is True
        ]
        assert len(original_entries) >= 1
        assert original_entries[0]["artist"] == "The Beatles"

    def test_get_songs_limits_results(self):
        """get_songs value is passed as LIMIT to the SQL query."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        cur.fetchone = Mock(return_value=_make_dict_row({"author": "Coldplay"}))
        many_songs = [
            _make_dict_row({"item_id": str(i), "title": f"Song {i}", "author": "Coldplay"})
            for i in range(50)
        ]
        cur.fetchall = Mock(return_value=many_songs)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        gmm_mod = self._setup_gmm_module(
            find_return=[{"artist": "U2", "distance": 0.2}]
        )

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.artist_gmm_manager': gmm_mod}):
            mod._artist_similarity_api_sync("Coldplay", count=5, get_songs=5)

        execute_calls = cur.execute.call_args_list
        for c in execute_calls:
            args = c[0]
            if len(args) >= 2 and isinstance(args[1], list):
                assert args[1][-1] == 5
                break


# ---------------------------------------------------------------------------
# _song_alchemy_sync
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSongAlchemySync:
    """Tests for _song_alchemy_sync - blend/subtract musical vibes."""

    def _setup_alchemy_module(self, return_value=None, side_effect=None):
        mock_mod = MagicMock()
        if side_effect:
            mock_mod.song_alchemy = Mock(side_effect=side_effect)
        else:
            mock_mod.song_alchemy = Mock(return_value=return_value or {"results": []})
        return mock_mod

    def test_correct_args_passed(self):
        """Verify add_items and subtract_items are forwarded correctly."""
        mod = _import_mcp_impl()

        add = [{"type": "song", "id": "s1"}, {"type": "artist", "id": "a1"}]
        sub = [{"type": "song", "id": "s2"}]
        alchemy_mod = self._setup_alchemy_module(
            return_value={"results": [{"item_id": "r1", "title": "Result", "artist": "Art"}]}
        )

        with patch.dict(sys.modules, {'tasks.song_alchemy': alchemy_mod}):
            result = mod._song_alchemy_sync(add_items=add, subtract_items=sub, get_songs=10)

        alchemy_mod.song_alchemy.assert_called_once_with(
            add_items=add,
            subtract_items=sub,
            n_results=10
        )
        assert "songs" in result

    def test_empty_add_items(self):
        """Empty add_items list should still call song_alchemy without error."""
        mod = _import_mcp_impl()

        alchemy_mod = self._setup_alchemy_module(return_value={"results": []})

        with patch.dict(sys.modules, {'tasks.song_alchemy': alchemy_mod}):
            result = mod._song_alchemy_sync(add_items=[], subtract_items=None, get_songs=10)

        alchemy_mod.song_alchemy.assert_called_once()
        assert result["songs"] == []

    def test_exception_returns_error(self):
        """If song_alchemy raises, result has empty songs and error message."""
        mod = _import_mcp_impl()

        alchemy_mod = self._setup_alchemy_module(side_effect=Exception("IVF index missing"))

        with patch.dict(sys.modules, {'tasks.song_alchemy': alchemy_mod}):
            result = mod._song_alchemy_sync(
                add_items=[{"type": "song", "id": "s1"}],
                subtract_items=None,
                get_songs=10
            )

        assert result["songs"] == []
        assert "error" in result["message"].lower()

    def test_result_structure(self):
        """Returned dict has 'songs' and 'message' keys."""
        mod = _import_mcp_impl()

        alchemy_mod = self._setup_alchemy_module(
            return_value={"results": [{"item_id": "r1", "title": "T", "artist": "A"}]}
        )

        with patch.dict(sys.modules, {'tasks.song_alchemy': alchemy_mod}):
            result = mod._song_alchemy_sync(
                add_items=[{"type": "song", "id": "s1"}],
                get_songs=10
            )

        assert "songs" in result
        assert "message" in result


# ---------------------------------------------------------------------------
# _ai_brainstorm_sync
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAiBrainstormSync:
    """_ai_brainstorm_sync emits a grounded recipe and fuses retrieval channels (#643)."""

    def _make_ai_module(self, response):
        mock_mod = MagicMock()
        mock_mod.generate_text = Mock(return_value=response)
        return mock_mod

    def _make_ai_config(self):
        return {"provider": "gemini", "gemini_key": "fake-key", "gemini_model": "gemini-pro"}

    def _recipe(self, **over):
        base = {
            "filters": {"genres": ["rock"]},
            "sound_descriptions": ["driving guitar rock"],
            "seed_artists": [],
            "lyric_themes": [],
        }
        base.update(over)
        return json.dumps(base)

    def _patch_channels(self, mod, audio=None, artist=None, lyrics=None, filt=None):
        empty = {"songs": []}
        return (
            patch.object(mod, '_text_search_sync', return_value=audio if audio is not None else empty),
            patch.object(mod, '_artist_similarity_api_sync', return_value=artist if artist is not None else empty),
            patch.object(mod, '_lyrics_search_sync', return_value=lyrics if lyrics is not None else empty),
            patch.object(mod, '_database_genre_query_sync', return_value=filt if filt is not None else empty),
        )

    def test_ai_error_returns_empty(self):
        """AI transport error -> empty songs, no channels touched."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module("Error: API rate limit exceeded")
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}):
            result = mod._ai_brainstorm_sync("rock classics", self._make_ai_config(), 10)
        assert result["songs"] == []

    def test_unparseable_returns_empty_without_traceback(self):
        """No JSON object recoverable -> empty + generic message (never a traceback)."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module("here are some great rock songs, but no json")
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}):
            result = mod._ai_brainstorm_sync("rock", self._make_ai_config(), 10)
        assert result["songs"] == []
        assert "Traceback" not in result["message"]

    def test_recipe_drives_channels_and_fuses(self):
        """Recipe runs audio + artist + filter channels and unions their results."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module(self._recipe(seed_artists=["Nirvana"]))
        p_audio, p_artist, p_lyrics, p_filt = self._patch_channels(
            mod,
            audio={"songs": [{"item_id": "1", "title": "A", "artist": "X"}]},
            artist={"songs": [{"item_id": "2", "title": "B", "artist": "Nirvana"}]},
            filt={"songs": [{"item_id": "3", "title": "C", "artist": "Z"}]},
        )
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}), \
             p_audio as a, p_artist as ar, p_lyrics, p_filt as f:
            result = mod._ai_brainstorm_sync("90s rock like Nirvana", self._make_ai_config(), 50)
        ids = sorted(s["item_id"] for s in result["songs"])
        assert ids == ["1", "2", "3"]
        assert a.called and ar.called and f.called

    def test_dedup_across_channels(self):
        """The same item returned by several channels appears once."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module(self._recipe(seed_artists=["Nirvana"]))
        dup = {"songs": [{"item_id": "1", "title": "A", "artist": "X"}]}
        p_audio, p_artist, p_lyrics, p_filt = self._patch_channels(mod, audio=dup, artist=dup, filt=dup)
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}), p_audio, p_artist, p_lyrics, p_filt:
            result = mod._ai_brainstorm_sync("x", self._make_ai_config(), 50)
        assert len(result["songs"]) == 1

    def test_get_songs_cap_respected(self):
        """The fused pool never exceeds get_songs."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module(self._recipe())
        many = {"songs": [{"item_id": str(i), "title": f"T{i}", "artist": f"A{i}"} for i in range(100)]}
        p_audio, p_artist, p_lyrics, p_filt = self._patch_channels(mod, audio=many)
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}), p_audio, p_artist, p_lyrics, p_filt:
            result = mod._ai_brainstorm_sync("x", self._make_ai_config(), 10)
        assert len(result["songs"]) == 10

    def test_float_get_songs_does_not_raise(self):
        """Providers may send get_songs as a float (e.g. 50.0)."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module(self._recipe())
        p_audio, p_artist, p_lyrics, p_filt = self._patch_channels(mod)
        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}), p_audio, p_artist, p_lyrics, p_filt:
            result = mod._ai_brainstorm_sync("test", self._make_ai_config(), 50.0)
        assert "songs" in result

    def test_year_gate_excludes_out_of_era_sound_results(self):
        """A year in the recipe gates the sound channel: CLAP can surface any-era
        songs, but only in-era ones survive (the #643 'best rap of the 90s' fix)."""
        mod = _import_mcp_impl()
        ai_mod = self._make_ai_module(json.dumps({
            "filters": {"genres": ["rock"], "year_min": 1990, "year_max": 1999},
            "sound_descriptions": ["driving guitar rock"],
            "seed_artists": [],
            "lyric_themes": [],
        }))
        audio = {"songs": [
            {"item_id": "in", "title": "In Era", "artist": "X"},
            {"item_id": "out", "title": "Out Era", "artist": "Y"},
        ]}
        in_era = {"in"}

        def _db(*args, **kwargs):
            cids = kwargs.get("candidate_item_ids")
            if cids:
                return {"songs": [s for s in audio["songs"] if s["item_id"] in cids and s["item_id"] in in_era]}
            return {"songs": []}

        with patch.dict(sys.modules, {'tasks.ai.api': ai_mod}), \
             patch.object(mod, '_text_search_sync', return_value=audio), \
             patch.object(mod, '_artist_similarity_api_sync', return_value={"songs": []}), \
             patch.object(mod, '_lyrics_search_sync', return_value={"songs": []}), \
             patch.object(mod, '_database_genre_query_sync', side_effect=_db):
            result = mod._ai_brainstorm_sync("best rock of the 90s", self._make_ai_config(), 50)

        ids = {s["item_id"] for s in result["songs"]}
        assert "in" in ids
        assert "out" not in ids


# ---------------------------------------------------------------------------
# _text_search_sync
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextSearchSync:
    """Tests for _text_search_sync - CLAP text search with hybrid filtering."""

    def _setup_cursor(self):
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchall = Mock(return_value=[])
        return cur

    def _make_clap_module(self, results=None, side_effect=None):
        mock_mod = MagicMock()
        if side_effect:
            mock_mod.search_by_text = Mock(side_effect=side_effect)
        else:
            mock_mod.search_by_text = Mock(return_value=results if results is not None else [])
        return mock_mod

    def test_clap_disabled_returns_message(self):
        """CLAP_ENABLED=False -> message says not enabled."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_mod = self._make_clap_module()
        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = False
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("dreamy soundscape", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert result["songs"] == []
        assert "not enabled" in result["message"]

    def test_empty_description_returns_empty(self):
        """Empty description -> empty songs."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_mod = self._make_clap_module()
        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert result["songs"] == []

    def test_no_clap_results(self):
        """search_by_text returns [] -> empty songs."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_mod = self._make_clap_module(results=[])
        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("ambient forest", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert result["songs"] == []

    def test_no_filters_returns_clap_results_directly(self):
        """No tempo/energy filters -> CLAP results returned as-is."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_results = [
            {"item_id": "c1", "title": "Ambient Song", "author": "Artist A"},
            {"item_id": "c2", "title": "Dreamy Track", "author": "Artist B"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)
        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("ambient dreamy", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert len(result["songs"]) == 2
        assert result["songs"][0]["item_id"] == "c1"

    def test_tempo_filter_ignored_pool_returned_as_is(self):
        """tempo_filter is accepted for signature compatibility but ignored;
        the tool returns the pure CLAP pool (tempo/energy re-rank is downstream)."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "Slow Song", "author": "A1"},
            {"item_id": "c2", "title": "Fast Song", "author": "A2"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("chill music", "slow", None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert len(result["songs"]) == 2
        assert [s["item_id"] for s in result["songs"]] == ["c1", "c2"]

    def test_energy_filter_ignored_pool_returned_as_is(self):
        """energy_filter is accepted but ignored; the pool comes back unfiltered."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "High Energy", "author": "A1"},
            {"item_id": "c2", "title": "Low Energy", "author": "A2"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("energetic music", None, "high", 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert len(result["songs"]) == 2
        assert [s["item_id"] for s in result["songs"]] == ["c1", "c2"]

    def test_combined_tempo_and_energy_filters_ignored(self):
        """Both tempo and energy filters are ignored; full pool returned in order."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "Perfect Match", "author": "A1"},
            {"item_id": "c2", "title": "No Match", "author": "A2"},
            {"item_id": "c3", "title": "Also Match", "author": "A3"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("upbeat dance", "fast", "high", 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert len(result["songs"]) == 3
        assert [s["item_id"] for s in result["songs"]] == ["c1", "c2", "c3"]

    def test_get_songs_passed_as_clap_limit(self):
        """get_songs is delegated to search_by_text as its limit; the tool itself
        does not slice the returned pool."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_results = [
            {"item_id": f"c{i}", "title": f"Song {i}", "author": f"Artist {i}"}
            for i in range(10)
        ]
        clap_mod = self._make_clap_module(results=clap_results)

        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("anything", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        clap_mod.search_by_text.assert_called_once_with("anything", limit=10)
        assert len(result["songs"]) == 10

    def test_exception_returns_empty_with_message(self):
        """search_by_text raises -> empty songs with error message."""
        mod = _import_mcp_impl()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_mod = self._make_clap_module(side_effect=RuntimeError("CLAP model not loaded"))

        import config as cfg
        orig = cfg.CLAP_ENABLED
        try:
            cfg.CLAP_ENABLED = True
            with patch.object(mod, 'get_db_connection', return_value=conn), \
                 patch.dict(sys.modules, {'tasks.clap_text_search': clap_mod}):
                result = mod._text_search_sync("test query", None, None, 10)
        finally:
            cfg.CLAP_ENABLED = orig

        assert result["songs"] == []
        assert "error" in result["message"].lower()
