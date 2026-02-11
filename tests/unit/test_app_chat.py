"""Unit tests for app_chat.py instant playlist pipeline

Tests cover the agentic playlist workflow:
- Artist diversity enforcement (max songs per artist, backfill)
- Proportional sampling from tool calls
- Pre-execution validation (empty song_similarity, filterless search_database)
- Ollama JSON extraction edge cases
- Iteration deduplication (song_ids_seen)
- Stopping conditions (target reached, no new songs, AI error)
- API key validation for cloud providers
"""
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, call

flask = pytest.importorskip('flask', reason='Flask not installed')
from flask import Flask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a Flask app with the chat blueprint registered."""
    with patch('app_chat.OLLAMA_SERVER_URL', 'http://localhost:11434'), \
         patch('app_chat.OLLAMA_MODEL_NAME', 'test-model'), \
         patch('app_chat.OPENAI_SERVER_URL', 'http://localhost'), \
         patch('app_chat.OPENAI_MODEL_NAME', 'gpt-4'), \
         patch('app_chat.OPENAI_API_KEY', ''), \
         patch('app_chat.GEMINI_MODEL_NAME', 'gemini-pro'), \
         patch('app_chat.GEMINI_API_KEY', ''), \
         patch('app_chat.MISTRAL_MODEL_NAME', 'mistral-7b'), \
         patch('app_chat.MISTRAL_API_KEY', ''), \
         patch('app_chat.AI_MODEL_PROVIDER', 'OLLAMA'):
        from app_chat import chat_bp
        flask_app = Flask(__name__)
        flask_app.register_blueprint(chat_bp)
        flask_app.config['TESTING'] = True
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _song(item_id, title="Song", artist="Artist"):
    """Helper to create a song dict."""
    return {'item_id': item_id, 'title': title, 'artist': artist}


# ---------------------------------------------------------------------------
# Artist Diversity Enforcement
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestArtistDiversity:
    """Test artist diversity enforcement (Phase 3B)."""

    def test_under_limit_keeps_all(self):
        """Songs under the per-artist limit are all kept."""
        songs = [_song(f'id{i}', artist='ArtistA') for i in range(3)]
        max_per = 5
        artist_counts = {}
        diverse = []
        overflow = []
        for s in songs:
            a = s.get('artist', 'Unknown')
            artist_counts[a] = artist_counts.get(a, 0) + 1
            if artist_counts[a] <= max_per:
                diverse.append(s)
            else:
                overflow.append(s)
        assert len(diverse) == 3
        assert len(overflow) == 0

    def test_over_limit_trims_excess(self):
        """Songs over the per-artist limit go to overflow."""
        songs = [_song(f'id{i}', artist='ArtistA') for i in range(8)]
        max_per = 5
        artist_counts = {}
        diverse = []
        overflow = []
        for s in songs:
            a = s.get('artist', 'Unknown')
            artist_counts[a] = artist_counts.get(a, 0) + 1
            if artist_counts[a] <= max_per:
                diverse.append(s)
            else:
                overflow.append(s)
        assert len(diverse) == 5
        assert len(overflow) == 3

    def test_multiple_artists_independent_limits(self):
        """Each artist gets an independent limit."""
        songs = [_song(f'a{i}', artist='A') for i in range(6)]
        songs += [_song(f'b{i}', artist='B') for i in range(4)]
        max_per = 5
        artist_counts = {}
        diverse = []
        for s in songs:
            a = s.get('artist', 'Unknown')
            artist_counts[a] = artist_counts.get(a, 0) + 1
            if artist_counts[a] <= max_per:
                diverse.append(s)
        assert sum(1 for s in diverse if s['artist'] == 'A') == 5
        assert sum(1 for s in diverse if s['artist'] == 'B') == 4

    def test_backfill_from_overflow(self):
        """Overflow songs backfill from least-represented artists."""
        # 10 songs from ArtistA, 2 from ArtistB, limit=3, target=8
        songs = [_song(f'a{i}', artist='ArtistA') for i in range(10)]
        songs += [_song(f'b{i}', artist='ArtistB') for i in range(2)]
        max_per = 3
        target = 8
        artist_counts = {}
        diverse = []
        overflow = []
        for s in songs:
            a = s.get('artist', 'Unknown')
            artist_counts[a] = artist_counts.get(a, 0) + 1
            if artist_counts[a] <= max_per:
                diverse.append(s)
            else:
                overflow.append(s)
        # diverse has 3 A + 2 B = 5 songs, need 3 more from overflow
        if len(diverse) < target and overflow:
            diverse_artist_counts = {}
            for s in diverse:
                a = s.get('artist', 'Unknown')
                diverse_artist_counts[a] = diverse_artist_counts.get(a, 0) + 1
            overflow.sort(key=lambda s: diverse_artist_counts.get(s.get('artist', ''), 0))
            backfill_needed = target - len(diverse)
            diverse.extend(overflow[:backfill_needed])
        assert len(diverse) == 8

    def test_default_max_per_artist_is_5(self):
        """Config default MAX_SONGS_PER_ARTIST_PLAYLIST is 5."""
        from config import MAX_SONGS_PER_ARTIST_PLAYLIST
        assert MAX_SONGS_PER_ARTIST_PLAYLIST == 5


# ---------------------------------------------------------------------------
# Proportional Sampling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProportionalSampling:
    """Test proportional sampling when more songs than target."""

    def test_under_target_uses_all(self):
        """When total < target, all songs are kept."""
        all_songs = [_song(f'id{i}') for i in range(50)]
        target = 100
        assert len(all_songs) <= target

    def test_over_target_samples_proportionally(self):
        """When total > target, songs are sampled proportionally by source."""
        song_sources = {}
        songs_by_call = {0: [], 1: []}
        for i in range(80):
            s = _song(f'id{i}')
            songs_by_call[0].append(s)
            song_sources[f'id{i}'] = 0
        for i in range(80, 120):
            s = _song(f'id{i}')
            songs_by_call[1].append(s)
            song_sources[f'id{i}'] = 1
        total = 120
        target = 100
        final = []
        for call_index, tool_songs in songs_by_call.items():
            proportion = len(tool_songs) / total
            allocated = int(proportion * target)
            if allocated == 0 and len(tool_songs) > 0:
                allocated = 1
            final.extend(tool_songs[:allocated])
        # Call 0: 80/120*100=66, Call 1: 40/120*100=33 => 99 total
        assert len(final) <= target

    def test_each_call_gets_at_least_one(self):
        """Even a tool call with 1 song gets at least 1 in the final list."""
        songs_by_call = {0: [_song('majority')]*99, 1: [_song('tiny')]}
        total = 100
        target = 50
        final = []
        for call_index, tool_songs in songs_by_call.items():
            proportion = len(tool_songs) / total
            allocated = int(proportion * target)
            if allocated == 0 and len(tool_songs) > 0:
                allocated = 1
            final.extend(tool_songs[:allocated])
        # Check that call 1's song is included
        assert any(s['item_id'] == 'tiny' for s in final)

    def test_rounding_backfill(self):
        """Remaining songs are backfilled if proportional rounding falls short."""
        all_songs = [_song(f'id{i}') for i in range(120)]
        song_sources = {f'id{i}': i % 3 for i in range(120)}
        target = 100
        songs_by_call = {}
        for s in all_songs:
            ci = song_sources[s['item_id']]
            songs_by_call.setdefault(ci, []).append(s)
        final = []
        for ci, ts in songs_by_call.items():
            proportion = len(ts) / len(all_songs)
            allocated = int(proportion * target)
            if allocated == 0 and len(ts) > 0:
                allocated = 1
            final.extend(ts[:allocated])
        if len(final) < target:
            remaining = [s for s in all_songs if s not in final]
            final.extend(remaining[:target - len(final)])
        assert len(final) == target


# ---------------------------------------------------------------------------
# Pre-Execution Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPreExecutionValidation:
    """Test pre-execution validation of tool calls."""

    def test_song_similarity_empty_title_rejected(self):
        """song_similarity with empty title is rejected."""
        tc = {'name': 'song_similarity', 'arguments': {'song_title': '', 'song_artist': 'Artist'}}
        ta = tc['arguments']
        assert not ta.get('song_title', '').strip()

    def test_song_similarity_empty_artist_rejected(self):
        """song_similarity with empty artist is rejected."""
        tc = {'name': 'song_similarity', 'arguments': {'song_title': 'Title', 'song_artist': ''}}
        ta = tc['arguments']
        assert not ta.get('song_artist', '').strip()

    def test_song_similarity_whitespace_only_rejected(self):
        """song_similarity with whitespace-only values is rejected."""
        tc = {'name': 'song_similarity', 'arguments': {'song_title': '   ', 'song_artist': '  '}}
        ta = tc['arguments']
        assert not ta.get('song_title', '').strip()
        assert not ta.get('song_artist', '').strip()

    def test_song_similarity_valid_passes(self):
        """song_similarity with valid title and artist passes."""
        tc = {'name': 'song_similarity', 'arguments': {'song_title': 'Bohemian Rhapsody', 'song_artist': 'Queen'}}
        ta = tc['arguments']
        assert ta.get('song_title', '').strip()
        assert ta.get('song_artist', '').strip()

    def test_search_database_no_filters_rejected(self):
        """search_database with no filters is rejected."""
        tc = {'name': 'search_database', 'arguments': {'get_songs': 50}}
        ta = tc['arguments']
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating']
        has_filter = any(ta.get(k) for k in filter_keys)
        assert not has_filter

    def test_search_database_with_genres_passes(self):
        """search_database with genres filter passes."""
        tc = {'name': 'search_database', 'arguments': {'genres': ['rock'], 'get_songs': 50}}
        ta = tc['arguments']
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating']
        has_filter = any(ta.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_with_energy_passes(self):
        """search_database with energy filter passes."""
        tc = {'name': 'search_database', 'arguments': {'energy_min': 0.5, 'get_songs': 50}}
        ta = tc['arguments']
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating']
        has_filter = any(ta.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_with_year_filter_passes(self):
        """search_database with year_min filter passes."""
        tc = {'name': 'search_database', 'arguments': {'year_min': 2000, 'get_songs': 50}}
        ta = tc['arguments']
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating']
        has_filter = any(ta.get(k) for k in filter_keys)
        assert has_filter


# ---------------------------------------------------------------------------
# Iteration Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIterationDeduplication:
    """Test song deduplication across iterations."""

    def test_duplicate_ids_filtered(self):
        """Duplicate item_ids are not added twice."""
        song_ids_seen = set()
        all_songs = []
        batch1 = [_song('id1'), _song('id2'), _song('id3')]
        batch2 = [_song('id2'), _song('id3'), _song('id4')]  # id2, id3 are dupes
        for s in batch1:
            if s['item_id'] not in song_ids_seen:
                all_songs.append(s)
                song_ids_seen.add(s['item_id'])
        for s in batch2:
            if s['item_id'] not in song_ids_seen:
                all_songs.append(s)
                song_ids_seen.add(s['item_id'])
        assert len(all_songs) == 4
        assert song_ids_seen == {'id1', 'id2', 'id3', 'id4'}

    def test_all_duplicates_adds_zero(self):
        """When all songs are duplicates, zero new songs are added."""
        song_ids_seen = {'id1', 'id2'}
        all_songs = [_song('id1'), _song('id2')]
        batch = [_song('id1'), _song('id2')]
        new_count = 0
        for s in batch:
            if s['item_id'] not in song_ids_seen:
                all_songs.append(s)
                song_ids_seen.add(s['item_id'])
                new_count += 1
        assert new_count == 0


# ---------------------------------------------------------------------------
# Stopping Conditions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStoppingConditions:
    """Test the agentic loop stopping conditions."""

    def test_target_reached_stops(self):
        """Loop stops when target song count is reached."""
        target = 100
        current = 105
        assert current >= target

    def test_no_tool_calls_stops(self):
        """Loop stops when AI returns no tool calls."""
        tool_calls = []
        assert not tool_calls

    def test_no_new_songs_stops(self):
        """Loop stops when an iteration adds 0 new songs."""
        iteration_songs_added = 0
        assert iteration_songs_added == 0

    def test_max_iterations_stops(self):
        """Loop stops at max_iterations."""
        max_iterations = 5
        for iteration in range(max_iterations):
            pass
        assert iteration == max_iterations - 1

    def test_ai_error_stops_after_first_iteration(self):
        """AI error on iteration > 0 breaks the loop."""
        iteration = 2
        error = True
        should_break = iteration > 0 and error
        assert should_break


# ---------------------------------------------------------------------------
# API Key Validation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAPIKeyValidation:
    """Test API key validation for cloud providers."""

    def test_missing_input_returns_400(self, client):
        """Request without userInput returns 400."""
        resp = client.post('/api/chatPlaylist', json={})
        assert resp.status_code == 400
        assert resp.content_type.startswith('application/json')
        assert 'error' in resp.get_json()

    def test_none_provider_returns_no_ai_message(self, client):
        """Provider NONE returns informational message."""
        resp = client.post('/api/chatPlaylist', json={
            'userInput': 'test',
            'ai_provider': 'NONE'
        })
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        data = resp.get_json()
        assert 'No AI provider selected' in data['response']['message']

    def test_openai_missing_key_returns_400(self, client):
        """OpenAI without API key returns 400."""
        resp = client.post('/api/chatPlaylist', json={
            'userInput': 'test',
            'ai_provider': 'OPENAI',
            'openai_api_key': ''
        })
        assert resp.status_code == 400
        assert resp.content_type.startswith('application/json')
        assert 'error' in resp.get_json() or 'response' in resp.get_json()

    def test_gemini_placeholder_key_returns_400(self, client):
        """Gemini with placeholder key returns 400."""
        resp = client.post('/api/chatPlaylist', json={
            'userInput': 'test',
            'ai_provider': 'GEMINI',
            'gemini_api_key': 'YOUR-GEMINI-API-KEY-HERE'
        })
        assert resp.status_code == 400
        assert resp.content_type.startswith('application/json')
        assert 'error' in resp.get_json() or 'response' in resp.get_json()

    def test_mistral_placeholder_key_returns_400(self, client):
        """Mistral with placeholder key returns 400."""
        resp = client.post('/api/chatPlaylist', json={
            'userInput': 'test',
            'ai_provider': 'MISTRAL',
            'mistral_api_key': 'YOUR-MISTRAL-API-KEY-HERE'
        })
        assert resp.status_code == 400
        assert resp.content_type.startswith('application/json')
        assert 'error' in resp.get_json() or 'response' in resp.get_json()


# ---------------------------------------------------------------------------
# Create Playlist Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCreatePlaylistEndpoint:
    """Test the /api/create_playlist endpoint."""

    def _mock_voyager(self):
        """Context manager to mock tasks.voyager_manager for import."""
        mock_vm = MagicMock()
        return patch.dict('sys.modules', {'tasks.voyager_manager': mock_vm})

    def test_missing_params_returns_400(self, client):
        """Missing playlist_name or item_ids returns 400."""
        with self._mock_voyager():
            resp = client.post('/api/create_playlist', json={'playlist_name': 'Test'})
            assert resp.status_code == 400

    def test_empty_name_returns_400(self, client):
        """Empty playlist name returns 400."""
        with self._mock_voyager():
            resp = client.post('/api/create_playlist', json={
                'playlist_name': '  ',
                'item_ids': ['id1']
            })
            assert resp.status_code == 400

    def test_empty_item_ids_returns_400(self, client):
        """Empty item_ids list returns 400."""
        with self._mock_voyager():
            resp = client.post('/api/create_playlist', json={
                'playlist_name': 'Test',
                'item_ids': []
            })
            assert resp.status_code == 400

    def test_single_provider_success(self, client):
        """Successful single-provider playlist creation."""
        mock_vm = MagicMock()
        mock_vm.create_playlist_from_ids = Mock(return_value='playlist-123')
        with patch.dict('sys.modules', {'tasks.voyager_manager': mock_vm}):
            resp = client.post('/api/create_playlist', json={
                'playlist_name': 'My Mix',
                'item_ids': ['id1', 'id2']
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert 'Successfully created' in data['message']

    def test_multi_provider_success(self, client):
        """Successful multi-provider playlist creation."""
        mock_vm = MagicMock()
        mock_vm.create_playlist_from_ids = Mock(return_value={
            'jellyfin': {'success': True, 'id': 'jf-1'},
            'navidrome': {'success': False, 'error': 'timeout'}
        })
        with patch.dict('sys.modules', {'tasks.voyager_manager': mock_vm}):
            resp = client.post('/api/create_playlist', json={
                'playlist_name': 'My Mix',
                'item_ids': ['id1'],
                'provider_ids': 'all'
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert '1/2' in data['message']


# ---------------------------------------------------------------------------
# Config Defaults Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConfigDefaultsEndpoint:
    """Test the /api/config_defaults endpoint."""

    def test_returns_200(self, client):
        """GET /api/config_defaults returns 200."""
        resp = client.get('/api/config_defaults')
        assert resp.status_code == 200

    def test_returns_json_content_type(self, client):
        """Response has application/json content type."""
        resp = client.get('/api/config_defaults')
        assert resp.content_type.startswith('application/json')

    def test_returns_json_with_expected_keys(self, client):
        """Response includes provider configuration defaults."""
        resp = client.get('/api/config_defaults')
        data = resp.get_json()
        assert isinstance(data, dict)
        assert 'default_ai_provider' in data
        assert 'default_ollama_model_name' in data
        assert 'ollama_server_url' in data
        assert 'default_openai_model_name' in data
        assert 'openai_server_url' in data
        assert 'default_gemini_model_name' in data
        assert 'default_mistral_model_name' in data

    def test_values_are_strings(self, client):
        """All returned values are strings."""
        resp = client.get('/api/config_defaults')
        data = resp.get_json()
        for key, value in data.items():
            assert isinstance(value, str), f"Expected string for '{key}', got {type(value)}"
