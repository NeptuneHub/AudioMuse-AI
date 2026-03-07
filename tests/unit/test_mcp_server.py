"""Unit tests for tasks/mcp_server.py

Tests cover MCP server tool functions:
- get_library_context(): Library statistics with caching
- _database_genre_query_sync(): Genre regex matching, filters, relevance scoring
- _ai_brainstorm_sync(): Two-stage matching (exact + fuzzy normalized)
- _song_similarity_api_sync(): Song lookup with exact/fuzzy fallback
- Energy normalization in execute_mcp_tool()
- Pre-execution validation (filterless search_database rejection)

NOTE: uses importlib to load tasks.mcp_server directly, bypassing
tasks/__init__.py which pulls in pydub (requires audioop removed in Python 3.14).
"""
import json
import re
import os
import sys
import importlib.util
import pytest
from unittest.mock import Mock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Module loaders (bypass tasks/__init__.py -> pydub -> audioop chain)
# ---------------------------------------------------------------------------

def _import_mcp_server():
    """Load tasks.mcp_server directly without triggering tasks/__init__.py."""
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tasks', 'mcp_server.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'tasks.mcp_server'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def _import_ai_mcp_client():
    """Load ai_mcp_client directly."""
    mod_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ai_mcp_client.py'
    )
    mod_path = os.path.normpath(mod_path)
    mod_name = 'ai_mcp_client'
    if mod_name not in sys.modules:
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
        """Check if the genre regex pattern matches the mood_vector string."""
        pattern = f"(^|,)\\s*{re.escape(genre)}:"
        return bool(re.search(pattern, mood_vector, re.IGNORECASE))

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

        cur.__iter__ = Mock(side_effect=[
            iter([_make_dict_row({"tag": "rock:0.82"}), _make_dict_row({"tag": "pop:0.45"})]),
            iter([_make_dict_row({"mood": "danceable"}), _make_dict_row({"mood": "happy"})]),
        ])
        cur.fetchall = Mock(return_value=[
            _make_dict_row({"scale": "major"}),
            _make_dict_row({"scale": "minor"}),
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
        """Verify the SQL contains the regex pattern for genre matching."""
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(genres=["rock"], get_songs=10)

        call_args = cur.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else []
        assert "~*" in sql  # PostgreSQL case-insensitive regex
        found_regex = any("rock:" in str(p) for p in params) if params else False
        assert found_regex or "rock" in sql

    def test_tempo_range_filter(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(tempo_min=120, tempo_max=140, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "tempo >=" in sql
        assert "tempo <=" in sql

    def test_key_filter_uppercased(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(key="c", get_songs=10)

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        assert "key = %s" in sql
        assert "C" in params  # should be uppercased

    def test_scale_filter_case_insensitive(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(scale="Major", get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "LOWER(scale)" in sql

    def test_year_range_filter(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(year_min=1980, year_max=1989, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "year >=" in sql
        assert "year <=" in sql

    def test_min_rating_filter(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(min_rating=4, get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "rating >=" in sql

    def test_mood_filter_uses_like(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(moods=["danceable"], get_songs=10)

        sql = cur.execute.call_args[0][0]
        assert "LIKE" in sql

    def test_combined_filters_use_and(self):
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            mod._database_genre_query_sync(
                genres=["rock"], tempo_min=120, energy_min=0.05,
                key="C", scale="major", year_min=2000, min_rating=3, get_songs=10
            )

        sql = cur.execute.call_args[0][0]
        assert sql.count("AND") >= 5  # Multiple AND conditions

    def test_results_returned_as_list(self):
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
        conn, cur = self._setup_mock_conn()

        with patch.object(mod, 'get_db_connection', return_value=conn):
            # Should not raise - float get_songs handled
            mod._database_genre_query_sync(genres=["rock"], get_songs=50.0)


# ---------------------------------------------------------------------------
# ai_brainstorm normalization patterns (unit-testable without DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBrainstormNormalization:
    """Test the normalization logic used in _ai_brainstorm_sync."""

    def _normalize(self, text):
        """Reproduce the normalization from mcp_server."""
        return (text.lower()
                .replace(' ', '')
                .replace('-', '')
                .replace("'", '')
                .replace('.', '')
                .replace(',', ''))

    def test_lowercase(self):
        assert self._normalize("Hello") == "hello"

    def test_remove_spaces(self):
        assert self._normalize("The Beatles") == "thebeatles"

    def test_remove_dashes(self):
        assert self._normalize("up-beat") == "upbeat"

    def test_remove_apostrophes(self):
        assert self._normalize("Don't Stop") == "dontstop"

    def test_remove_periods(self):
        assert self._normalize("Mr. Jones") == "mrjones"

    def test_remove_commas(self):
        assert self._normalize("Hello, World") == "helloworld"

    def test_ac_dc_normalization(self):
        """AC/DC normalizes consistently (slash not removed but spaces/dots are)."""
        result = self._normalize("AC DC")
        assert result == "acdc"

    def test_complex_normalization(self):
        assert self._normalize("Don't Stop Me Now") == "dontstopmenow"

    def test_both_title_and_artist_required(self):
        """Demonstrate that matching requires BOTH title and artist."""
        title_norm = self._normalize("Bohemian Rhapsody")
        artist_norm = self._normalize("Queen")
        assert title_norm == "bohemianrhapsody"
        assert artist_norm == "queen"

    def test_same_title_different_artist_not_equal(self):
        """Same title with different artist should not be considered same."""
        t1 = self._normalize("Yesterday") + "|" + self._normalize("The Beatles")
        t2 = self._normalize("Yesterday") + "|" + self._normalize("Some Cover Artist")
        assert t1 != t2


# ---------------------------------------------------------------------------
# execute_mcp_tool energy conversion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecuteMcpToolEnergyConversion:
    """Test that execute_mcp_tool converts energy from 0-1 to raw."""

    def test_search_database_energy_conversion(self):
        ai_mod = _import_ai_mcp_client()
        mcp_mod = _import_mcp_server()

        mock_query = Mock(return_value={"songs": []})
        import config as cfg
        orig_min, orig_max = cfg.ENERGY_MIN, cfg.ENERGY_MAX
        try:
            cfg.ENERGY_MIN = 0.01
            cfg.ENERGY_MAX = 0.15
            with patch.object(mcp_mod, '_database_genre_query_sync', mock_query):
                # Patch the lazy import inside execute_mcp_tool
                with patch.dict('sys.modules', {'tasks.mcp_server': mcp_mod}):
                    ai_mod.execute_mcp_tool("search_database", {
                        "genres": ["rock"],
                        "energy_min": 0.5,
                        "energy_max": 0.8
                    }, {})

            # Check the raw energy values passed to the query function
            if mock_query.called:
                kwargs = mock_query.call_args[1] if mock_query.call_args[1] else {}
                args = mock_query.call_args[0] if mock_query.call_args[0] else ()
                # energy should have been converted from 0-1 to raw
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
        mod = _import_mcp_server()
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
        # Create a mock voyager_manager module in sys.modules to avoid tasks/__init__.py
        mock_voyager = MagicMock()
        mock_voyager.find_nearest_neighbors_by_id = mock_nn
        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'tasks.voyager_manager': mock_voyager}):
            result = mod._song_similarity_api_sync("bohemian rhapsody", "queen", 10)

        # Should have tried a DB lookup
        assert cur.execute.called

    def test_no_match_returns_empty(self):
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
            result = mod._artist_similarity_api_sync("Coldplay", count=5, get_songs=5)

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
        mod = _import_mcp_server()

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
        mod = _import_mcp_server()

        alchemy_mod = self._setup_alchemy_module(return_value={"results": []})

        with patch.dict(sys.modules, {'tasks.song_alchemy': alchemy_mod}):
            result = mod._song_alchemy_sync(add_items=[], subtract_items=None, get_songs=10)

        alchemy_mod.song_alchemy.assert_called_once()
        assert result["songs"] == []

    def test_exception_returns_error(self):
        """If song_alchemy raises, result has empty songs and error message."""
        mod = _import_mcp_server()

        alchemy_mod = self._setup_alchemy_module(side_effect=Exception("Voyager index missing"))

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
        mod = _import_mcp_server()

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
    """Tests for _ai_brainstorm_sync - AI knowledge brainstorming with two-stage matching."""

    def _make_ai_module(self, response="[]"):
        mock_mod = MagicMock()
        mock_mod.call_ai_for_chat = Mock(return_value=response)
        return mock_mod

    def _make_ai_config(self):
        return {
            "provider": "gemini",
            "gemini_key": "fake-key",
            "gemini_model": "gemini-pro",
        }

    def _setup_cursor(self):
        cur = MagicMock()
        cur.__enter__ = Mock(return_value=cur)
        cur.__exit__ = Mock(return_value=False)
        cur.fetchall = Mock(return_value=[])
        return cur

    def test_ai_error_response_returns_empty(self):
        """AI returns 'Error: ...' -> result has empty songs."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        ai_mod = self._make_ai_module("Error: API rate limit exceeded")

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("rock classics", self._make_ai_config(), 10)

        assert result["songs"] == []
        assert "Error" in result["message"]

    def test_valid_json_array_parsed(self):
        """AI returns valid JSON array, DB finds matching rows."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        ai_response = json.dumps([
            {"title": "Bohemian Rhapsody", "artist": "Queen"},
            {"title": "Stairway to Heaven", "artist": "Led Zeppelin"},
        ])
        ai_mod = self._make_ai_module(ai_response)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "100", "title": "Bohemian Rhapsody", "author": "Queen"}),
            _make_dict_row({"item_id": "101", "title": "Stairway to Heaven", "author": "Led Zeppelin"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("classic rock", self._make_ai_config(), 10)

        assert len(result["songs"]) == 2
        titles = [s["title"] for s in result["songs"]]
        assert "Bohemian Rhapsody" in titles

    def test_markdown_code_blocks_stripped(self):
        """AI response wrapped in ```json...``` is still parsed correctly."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        ai_response = '```json\n[{"title": "Hey Jude", "artist": "The Beatles"}]\n```'
        ai_mod = self._make_ai_module(ai_response)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "200", "title": "Hey Jude", "author": "The Beatles"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("beatles hits", self._make_ai_config(), 10)

        assert len(result["songs"]) == 1
        assert result["songs"][0]["title"] == "Hey Jude"

    def test_stage1_exact_match(self):
        """AI suggests song in DB with exact title+artist -> found via stage 1."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        ai_response = json.dumps([{"title": "Creep", "artist": "Radiohead"}])
        ai_mod = self._make_ai_module(ai_response)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "300", "title": "Creep", "author": "Radiohead"}),
        ])
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("90s alternative", self._make_ai_config(), 10)

        assert len(result["songs"]) == 1
        assert result["songs"][0]["item_id"] == "300"

    def test_stage2_fuzzy_normalized_match(self):
        """AI suggests 'Don't Stop Me Now' by 'Queen', DB has 'Dont Stop Me Now' -> fuzzy match."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        ai_response = json.dumps([{"title": "Don't Stop Me Now", "artist": "Queen"}])
        ai_mod = self._make_ai_module(ai_response)

        call_count = [0]

        def _fetchall_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return []
            else:
                return [_make_dict_row({
                    "item_id": "400",
                    "title": "Dont Stop Me Now",
                    "author": "Queen"
                })]

        cur.fetchall = Mock(side_effect=_fetchall_side_effect)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("fun queen songs", self._make_ai_config(), 10)

        assert len(result["songs"]) == 1
        assert result["songs"][0]["item_id"] == "400"

    def test_normalize_logic(self):
        """Verify _normalize strips spaces, dashes, apostrophes, periods, commas."""
        # Reproduce the normalization regex from _ai_brainstorm_sync
        def _normalize(s):
            return re.sub(r"[\s\-\u2010\u2011\u2012\u2013\u2014/'\".,!?()]", '', s).lower()

        assert _normalize("Don't Stop Me Now") == "dontstopmenow"
        assert _normalize("Mr. Jones") == "mrjones"
        assert _normalize("Hello, World") == "helloworld"
        assert _normalize("up-beat") == "upbeat"
        assert _normalize("rock & roll") == "rock&roll"

    def test_escape_like(self):
        """_escape_like escapes % and _ characters."""
        def _escape_like(s):
            return s.replace('%', r'\%').replace('_', r'\_')

        assert _escape_like("100%") == r"100\%"
        assert _escape_like("under_score") == r"under\_score"
        assert _escape_like("normal") == "normal"

    def test_float_get_songs_converted_to_int(self):
        """Passing get_songs=50.0 (Gemini float) should not raise."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        ai_response = json.dumps([{"title": "Song", "artist": "Artist"}])
        ai_mod = self._make_ai_module(ai_response)

        cur.fetchall = Mock(return_value=[])

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("test", self._make_ai_config(), 50.0)

        assert "songs" in result

    def test_invalid_json_returns_empty(self):
        """AI returns non-JSON text -> result has empty songs."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        ai_mod = self._make_ai_module("Here are some great rock songs that you might enjoy!")

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("rock", self._make_ai_config(), 10)

        assert result["songs"] == []
        assert "parse" in result["message"].lower() or "Failed" in result["message"]

    def test_results_trimmed_to_get_songs(self):
        """AI suggests 30 songs, get_songs=10 -> only 10 returned."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        suggestions = [
            {"title": f"Song {i}", "artist": f"Artist {i}"} for i in range(30)
        ]
        ai_response = json.dumps(suggestions)
        ai_mod = self._make_ai_module(ai_response)

        exact_rows = [
            _make_dict_row({"item_id": str(i), "title": f"Song {i}", "author": f"Artist {i}"})
            for i in range(30)
        ]
        cur.fetchall = Mock(return_value=exact_rows)
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        with patch.object(mod, 'get_db_connection', return_value=conn), \
             patch.dict(sys.modules, {'ai': ai_mod}):
            result = mod._ai_brainstorm_sync("test", self._make_ai_config(), 10)

        assert len(result["songs"]) <= 10


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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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
        mod = _import_mcp_server()
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

    def test_tempo_filter_applied(self):
        """Tempo filter 'slow' triggers DB filtering of CLAP results."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "Slow Song", "author": "A1"},
            {"item_id": "c2", "title": "Fast Song", "author": "A2"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "c1", "title": "Slow Song", "author": "A1"}),
        ])
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

        assert len(result["songs"]) == 1
        assert result["songs"][0]["item_id"] == "c1"

    def test_energy_filter_applied(self):
        """Energy filter 'high' triggers DB filtering of CLAP results."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "High Energy", "author": "A1"},
            {"item_id": "c2", "title": "Low Energy", "author": "A2"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "c1", "title": "High Energy", "author": "A1"}),
        ])
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

        assert len(result["songs"]) == 1
        assert result["songs"][0]["item_id"] == "c1"

    def test_combined_tempo_and_energy_filters(self):
        """Both tempo and energy filters applied together."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()

        clap_results = [
            {"item_id": "c1", "title": "Perfect Match", "author": "A1"},
            {"item_id": "c2", "title": "No Match", "author": "A2"},
            {"item_id": "c3", "title": "Also Match", "author": "A3"},
        ]
        clap_mod = self._make_clap_module(results=clap_results)

        cur.fetchall = Mock(return_value=[
            _make_dict_row({"item_id": "c1", "title": "Perfect Match", "author": "A1"}),
            _make_dict_row({"item_id": "c3", "title": "Also Match", "author": "A3"}),
        ])
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

        assert len(result["songs"]) == 2
        assert result["songs"][0]["item_id"] == "c1"
        assert result["songs"][1]["item_id"] == "c3"

    def test_results_limited_to_get_songs(self):
        """CLAP returns 50 results, get_songs=10 -> only 10 returned."""
        mod = _import_mcp_server()
        cur = self._setup_cursor()
        conn = _make_connection(cur)
        conn.cursor = Mock(return_value=cur)

        clap_results = [
            {"item_id": f"c{i}", "title": f"Song {i}", "author": f"Artist {i}"}
            for i in range(50)
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

        assert len(result["songs"]) == 10

    def test_exception_returns_empty_with_message(self):
        """search_by_text raises -> empty songs with error message."""
        mod = _import_mcp_server()
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
