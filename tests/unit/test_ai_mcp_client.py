"""Unit tests for ai_mcp_client.py

Tests cover:
- _build_system_prompt(): Prompt generation with tool decision trees, library context
- get_mcp_tools(): Tool definitions based on CLAP_ENABLED
- execute_mcp_tool(): Tool dispatch with energy conversion, normalization
- call_ai_with_mcp_tools(): Provider dispatch routing
- _call_ollama_with_tools(): JSON parsing, fallbacks, timeouts
- _call_gemini_with_tools(): Gemini API mocking, schema conversion
- _call_openai_with_tools(): OpenAI API mocking, tool extraction
- _call_mistral_with_tools(): Mistral API mocking, key validation

NOTE: uses importlib via conftest.py ai_mcp_client_mod fixture to load
ai_mcp_client directly, bypassing tasks/__init__.py -> pydub -> audioop chain.

httpx and google.genai are not installed in the test environment, so we
install lightweight mock modules into sys.modules at import time.
"""
import json
import sys
import types
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Install stub modules for optional dependencies not present in test env
# ---------------------------------------------------------------------------

def _ensure_httpx_stub():
    """Install a lightweight httpx stub if httpx is not installed."""
    if 'httpx' in sys.modules and not isinstance(sys.modules['httpx'], types.ModuleType):
        return  # already a mock
    try:
        import httpx  # noqa: F401
    except ImportError:
        httpx_mod = types.ModuleType('httpx')

        class _ReadTimeout(Exception):
            pass

        class _TimeoutException(Exception):
            pass

        class _Client:
            def __init__(self, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def post(self, *a, **kw):
                raise NotImplementedError("stub")

        httpx_mod.ReadTimeout = _ReadTimeout
        httpx_mod.TimeoutException = _TimeoutException
        httpx_mod.Client = _Client
        sys.modules['httpx'] = httpx_mod


def _ensure_google_genai_stub():
    """Install google.genai stub if not installed."""
    try:
        import google.genai  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # Create the google package if needed
        if 'google' not in sys.modules:
            google_mod = types.ModuleType('google')
            google_mod.__path__ = []
            sys.modules['google'] = google_mod
        genai_mod = types.ModuleType('google.genai')
        genai_mod.Client = MagicMock
        genai_types = types.ModuleType('google.genai.types')
        genai_types.Tool = MagicMock
        genai_types.GenerateContentConfig = MagicMock
        genai_types.ToolConfig = MagicMock
        genai_types.FunctionCallingConfig = MagicMock
        genai_mod.types = genai_types
        sys.modules['google.genai'] = genai_mod
        sys.modules['google.genai.types'] = genai_types


_ensure_httpx_stub()
_ensure_google_genai_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_library_context(**overrides):
    """Build a library_context dict with sensible defaults."""
    ctx = {
        'total_songs': 500,
        'unique_artists': 80,
        'year_min': 1965,
        'year_max': 2024,
        'has_ratings': True,
        'rated_songs_pct': 40.0,
        'top_genres': ['rock', 'pop', 'metal', 'jazz', 'electronic'],
        'top_moods': ['danceable', 'aggressive', 'happy'],
        'scales': ['major', 'minor'],
    }
    ctx.update(overrides)
    return ctx


def _make_tools(include_text_search=True):
    """Build a minimal list of tool dicts for prompt building."""
    tools = [
        {'name': 'song_similarity', 'description': 'Find similar songs', 'inputSchema': {}},
        {'name': 'artist_similarity', 'description': 'Find artist songs', 'inputSchema': {}},
        {'name': 'song_alchemy', 'description': 'Blend artists', 'inputSchema': {}},
        {'name': 'ai_brainstorm', 'description': 'AI knowledge', 'inputSchema': {}},
        {'name': 'search_database', 'description': 'Search by filters', 'inputSchema': {}},
    ]
    if include_text_search:
        tools.insert(1, {'name': 'text_search', 'description': 'CLAP text search', 'inputSchema': {}})
    return tools


# ---------------------------------------------------------------------------
# TestBuildSystemPrompt
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildSystemPrompt:
    """Test _build_system_prompt() - pure logic, no network/DB."""

    def test_prompt_includes_tool_names(self, ai_mcp_client_mod):
        tools = _make_tools(include_text_search=True)
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        for t in tools:
            assert t['name'] in prompt

    def test_clap_decision_tree_has_six_steps(self, ai_mcp_client_mod):
        """With text_search present, decision tree should have 6 numbered steps."""
        tools = _make_tools(include_text_search=True)
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        # The decision tree section should contain step 6
        lines = prompt.split('\n')
        decision_lines = [l for l in lines if l.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.'))]
        assert any(l.strip().startswith('6.') for l in decision_lines)
        assert 'text_search' in prompt

    def test_no_clap_decision_tree_has_five_steps(self, ai_mcp_client_mod):
        """Without text_search, decision tree should have 5 steps, not 6."""
        tools = _make_tools(include_text_search=False)
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        # Extract only the TOOL SELECTION section lines (numbered decision tree)
        lines = prompt.split('\n')
        decision_lines = [l for l in lines
                          if l.strip() and l.strip()[0].isdigit()
                          and l.strip()[1] == '.'
                          and '->' in l]
        # Should have exactly 5 decision tree entries
        assert len(decision_lines) == 5
        # text_search should NOT appear as a decision tree target
        decision_text = '\n'.join(decision_lines)
        assert '-> text_search' not in decision_text

    def test_library_context_injected(self, ai_mcp_client_mod):
        ctx = _make_library_context()
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert '500 songs' in prompt
        assert '80 artists' in prompt

    def test_no_library_section_when_none(self, ai_mcp_client_mod):
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        assert "USER'S MUSIC LIBRARY" not in prompt

    def test_no_library_section_when_zero_songs(self, ai_mcp_client_mod):
        ctx = _make_library_context(total_songs=0)
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert "USER'S MUSIC LIBRARY" not in prompt

    def test_dynamic_genres_from_context(self, ai_mcp_client_mod):
        ctx = _make_library_context(top_genres=['synthwave', 'darkwave', 'ebm'])
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert 'synthwave' in prompt
        assert 'darkwave' in prompt

    def test_dynamic_moods_from_context(self, ai_mcp_client_mod):
        ctx = _make_library_context(top_moods=['melancholic', 'euphoric'])
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert 'melancholic' in prompt
        assert 'euphoric' in prompt

    def test_fallback_genres_when_no_context(self, ai_mcp_client_mod):
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        # Fallback genres from _FALLBACK_GENRES
        assert 'rock' in prompt
        assert 'jazz' in prompt
        assert 'electronic' in prompt

    def test_fallback_moods_when_no_context(self, ai_mcp_client_mod):
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, None)
        # Fallback moods from _FALLBACK_MOODS
        assert 'danceable' in prompt
        assert 'aggressive' in prompt

    def test_year_range_shown(self, ai_mcp_client_mod):
        ctx = _make_library_context(year_min=1980, year_max=2023)
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert '1980' in prompt
        assert '2023' in prompt

    def test_rating_info_shown_when_has_ratings(self, ai_mcp_client_mod):
        ctx = _make_library_context(has_ratings=True, rated_songs_pct=65.0)
        tools = _make_tools()
        prompt = ai_mcp_client_mod._build_system_prompt(tools, ctx)
        assert '65.0%' in prompt
        assert 'ratings' in prompt


# ---------------------------------------------------------------------------
# TestGetMcpTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetMcpTools:
    """Test get_mcp_tools() - tool definitions based on CLAP_ENABLED."""

    def test_returns_six_tools_with_clap(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        assert len(tools) == 6

    def test_returns_five_tools_without_clap(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = False
        tools = ai_mcp_client_mod.get_mcp_tools()
        assert len(tools) == 5

    def test_core_tool_names_present(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        names = [t['name'] for t in tools]
        for expected in ['song_similarity', 'artist_similarity', 'song_alchemy',
                         'ai_brainstorm', 'search_database']:
            assert expected in names

    def test_text_search_present_only_with_clap(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        names_clap = [t['name'] for t in ai_mcp_client_mod.get_mcp_tools()]
        assert 'text_search' in names_clap

        cfg.CLAP_ENABLED = False
        names_no_clap = [t['name'] for t in ai_mcp_client_mod.get_mcp_tools()]
        assert 'text_search' not in names_no_clap

    def test_tools_have_required_keys(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'inputSchema' in tool

    def test_song_similarity_requires_title_and_artist(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        ss = next(t for t in tools if t['name'] == 'song_similarity')
        required = ss['inputSchema'].get('required', [])
        assert 'song_title' in required
        assert 'song_artist' in required

    def test_search_database_has_filter_properties(self, ai_mcp_client_mod):
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        sd = next(t for t in tools if t['name'] == 'search_database')
        props = sd['inputSchema']['properties']
        for key in ['genres', 'moods', 'energy_min', 'energy_max',
                     'tempo_min', 'tempo_max', 'key', 'scale',
                     'year_min', 'year_max', 'min_rating']:
            assert key in props, f"Missing property: {key}"

    def test_priority_numbering_with_clap(self, ai_mcp_client_mod):
        """artist_similarity description says #3 when CLAP enabled."""
        import config as cfg
        cfg.CLAP_ENABLED = True
        tools = ai_mcp_client_mod.get_mcp_tools()
        artist_tool = next(t for t in tools if t['name'] == 'artist_similarity')
        assert '#3' in artist_tool['description']

    def test_priority_numbering_without_clap(self, ai_mcp_client_mod):
        """artist_similarity description says #2 when CLAP disabled."""
        import config as cfg
        cfg.CLAP_ENABLED = False
        tools = ai_mcp_client_mod.get_mcp_tools()
        artist_tool = next(t for t in tools if t['name'] == 'artist_similarity')
        assert '#2' in artist_tool['description']


# ---------------------------------------------------------------------------
# TestExecuteMcpTool
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecuteMcpTool:
    """Test execute_mcp_tool() - tool dispatch with energy conversion."""

    def _mock_mcp_server(self):
        """Create a mock mcp_server module with all required functions."""
        mock_mod = MagicMock()
        mock_mod._artist_similarity_api_sync = Mock(return_value={'songs': []})
        mock_mod._song_similarity_api_sync = Mock(return_value={'songs': []})
        mock_mod._database_genre_query_sync = Mock(return_value={'songs': []})
        mock_mod._ai_brainstorm_sync = Mock(return_value={'songs': []})
        mock_mod._song_alchemy_sync = Mock(return_value={'songs': []})
        mock_mod._text_search_sync = Mock(return_value={'songs': []})
        return mock_mod

    def test_energy_min_zero_maps_to_energy_min(self, ai_mcp_client_mod):
        import config as cfg
        cfg.ENERGY_MIN = 0.01
        cfg.ENERGY_MAX = 0.15
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('search_database', {
                'genres': ['rock'], 'energy_min': 0.0
            }, {})
        args = mock_mod._database_genre_query_sync.call_args[0]
        # args[5] is energy_min_raw
        assert abs(args[5] - 0.01) < 1e-9

    def test_energy_max_one_maps_to_energy_max(self, ai_mcp_client_mod):
        import config as cfg
        cfg.ENERGY_MIN = 0.01
        cfg.ENERGY_MAX = 0.15
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('search_database', {
                'genres': ['rock'], 'energy_max': 1.0
            }, {})
        args = mock_mod._database_genre_query_sync.call_args[0]
        # args[6] is energy_max_raw
        assert abs(args[6] - 0.15) < 1e-9

    def test_energy_mid_maps_to_midpoint(self, ai_mcp_client_mod):
        import config as cfg
        cfg.ENERGY_MIN = 0.01
        cfg.ENERGY_MAX = 0.15
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('search_database', {
                'genres': ['rock'], 'energy_min': 0.5
            }, {})
        args = mock_mod._database_genre_query_sync.call_args[0]
        assert abs(args[5] - 0.08) < 1e-9

    def test_no_energy_args_passes_none(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('search_database', {
                'genres': ['rock']
            }, {})
        args = mock_mod._database_genre_query_sync.call_args[0]
        # args[5]=energy_min_raw, args[6]=energy_max_raw should be None
        assert args[5] is None
        assert args[6] is None

    def test_unknown_tool_returns_error(self, ai_mcp_client_mod):
        # Must mock tasks.mcp_server to avoid pyaudioop import
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            result = ai_mcp_client_mod.execute_mcp_tool('nonexistent_tool', {}, {})
        assert 'error' in result

    def test_exception_returns_error(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        mock_mod._artist_similarity_api_sync.side_effect = RuntimeError("boom")
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            result = ai_mcp_client_mod.execute_mcp_tool('artist_similarity', {
                'artist': 'Test'
            }, {})
        assert 'error' in result

    def test_get_songs_defaults_to_100(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('artist_similarity', {
                'artist': 'Test'
            }, {})
        args = mock_mod._artist_similarity_api_sync.call_args[0]
        # args: (artist, count=15, get_songs)
        assert args[2] == 100  # default get_songs

    def test_song_alchemy_normalizes_string_items(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('song_alchemy', {
                'add_items': ['Metallica', 'Iron Maiden'],
                'subtract_items': ['Ballads']
            }, {})
        args = mock_mod._song_alchemy_sync.call_args[0]
        add_items = args[0]
        subtract_items = args[1]
        assert add_items == [
            {'type': 'artist', 'id': 'Metallica'},
            {'type': 'artist', 'id': 'Iron Maiden'}
        ]
        assert subtract_items == [{'type': 'artist', 'id': 'Ballads'}]

    def test_song_alchemy_handles_dict_items(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('song_alchemy', {
                'add_items': [{'type': 'artist', 'id': 'Metallica'}]
            }, {})
        args = mock_mod._song_alchemy_sync.call_args[0]
        assert args[0] == [{'type': 'artist', 'id': 'Metallica'}]

    def test_artist_similarity_hardcoded_count_15(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('artist_similarity', {
                'artist': 'Queen', 'get_songs': 50
            }, {})
        args = mock_mod._artist_similarity_api_sync.call_args[0]
        assert args[1] == 15  # hardcoded count

    def test_song_similarity_passes_title_and_artist(self, ai_mcp_client_mod):
        mock_mod = self._mock_mcp_server()
        with patch.dict('sys.modules', {'tasks.mcp_server': mock_mod}):
            ai_mcp_client_mod.execute_mcp_tool('song_similarity', {
                'song_title': 'Bohemian Rhapsody',
                'song_artist': 'Queen'
            }, {})
        args = mock_mod._song_similarity_api_sync.call_args[0]
        assert args[0] == 'Bohemian Rhapsody'
        assert args[1] == 'Queen'


# ---------------------------------------------------------------------------
# TestCallAiWithMcpTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallAiWithMcpTools:
    """Test call_ai_with_mcp_tools() - provider dispatch routing."""

    def test_dispatch_gemini(self, ai_mcp_client_mod):
        with patch.object(ai_mcp_client_mod, '_call_gemini_with_tools',
                          return_value={'tool_calls': []}) as mock_fn:
            result = ai_mcp_client_mod.call_ai_with_mcp_tools(
                'GEMINI', 'test', [], {}, [])
            mock_fn.assert_called_once()
        assert 'tool_calls' in result

    def test_dispatch_openai(self, ai_mcp_client_mod):
        with patch.object(ai_mcp_client_mod, '_call_openai_with_tools',
                          return_value={'tool_calls': []}) as mock_fn:
            result = ai_mcp_client_mod.call_ai_with_mcp_tools(
                'OPENAI', 'test', [], {}, [])
            mock_fn.assert_called_once()
        assert 'tool_calls' in result

    def test_dispatch_mistral(self, ai_mcp_client_mod):
        with patch.object(ai_mcp_client_mod, '_call_mistral_with_tools',
                          return_value={'tool_calls': []}) as mock_fn:
            result = ai_mcp_client_mod.call_ai_with_mcp_tools(
                'MISTRAL', 'test', [], {}, [])
            mock_fn.assert_called_once()
        assert 'tool_calls' in result

    def test_dispatch_ollama(self, ai_mcp_client_mod):
        with patch.object(ai_mcp_client_mod, '_call_ollama_with_tools',
                          return_value={'tool_calls': []}) as mock_fn:
            result = ai_mcp_client_mod.call_ai_with_mcp_tools(
                'OLLAMA', 'test', [], {}, [])
            mock_fn.assert_called_once()
        assert 'tool_calls' in result

    def test_unknown_provider_returns_error(self, ai_mcp_client_mod):
        result = ai_mcp_client_mod.call_ai_with_mcp_tools(
            'UNKNOWN', 'test', [], {}, [])
        assert 'error' in result
        assert 'Unsupported' in result['error']


# ---------------------------------------------------------------------------
# TestCallOllamaWithTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallOllamaWithTools:
    """Test _call_ollama_with_tools() - JSON parsing, fallbacks, timeouts."""

    def _make_httpx_client_mock(self, response_data):
        """Create a mock httpx.Client context manager returning given response."""
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = Mock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        return mock_client

    def _call(self, ai_mcp_client_mod, response_data, **kwargs):
        """Helper to call _call_ollama_with_tools with a mocked httpx.Client."""
        import httpx
        mock_client = self._make_httpx_client_mock(response_data)
        tools = _make_tools(include_text_search=False)
        ai_config = kwargs.get('ai_config', {'ollama_url': 'http://localhost:11434/api/generate',
                                              'ollama_model': 'llama3.1:8b'})
        log = []
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_ollama_with_tools(
                'test request', tools, ai_config, log)
        return result, log

    def test_valid_json_tool_calls_parsed(self, ai_mcp_client_mod):
        response_text = json.dumps({
            'tool_calls': [{'name': 'search_database', 'arguments': {'genres': ['rock']}}]
        })
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'tool_calls' in result
        assert len(result['tool_calls']) == 1
        assert result['tool_calls'][0]['name'] == 'search_database'

    def test_fallback_direct_array(self, ai_mcp_client_mod):
        response_text = json.dumps([
            {'name': 'search_database', 'arguments': {'genres': ['pop']}}
        ])
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['name'] == 'search_database'

    def test_fallback_single_object(self, ai_mcp_client_mod):
        response_text = json.dumps(
            {'name': 'artist_similarity', 'arguments': {'artist': 'Queen'}}
        )
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['name'] == 'artist_similarity'

    def test_markdown_code_block_stripping(self, ai_mcp_client_mod):
        inner = json.dumps({
            'tool_calls': [{'name': 'search_database', 'arguments': {'genres': ['jazz']}}]
        })
        response_text = f"```json\n{inner}\n```"
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'tool_calls' in result

    def test_schema_detection_returns_error(self, ai_mcp_client_mod):
        # JSON that looks like a schema: starts with '{', has '"type"' and '"array"'
        schema_response = json.dumps({
            'type': 'object',
            'properties': {'tool_calls': {'type': 'array', 'items': {}}}
        })
        result, _ = self._call(ai_mcp_client_mod, {'response': schema_response})
        assert 'error' in result
        assert 'schema' in result['error'].lower()

    def test_json_decode_error_returns_error(self, ai_mcp_client_mod):
        result, _ = self._call(ai_mcp_client_mod, {'response': 'not valid json {{'})
        assert 'error' in result
        assert 'Failed to parse' in result['error']

    def test_missing_arguments_defaults_to_empty(self, ai_mcp_client_mod):
        response_text = json.dumps({
            'tool_calls': [{'name': 'search_database'}]
        })
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['arguments'] == {}

    def test_invalid_tool_calls_skipped_all_invalid_returns_error(self, ai_mcp_client_mod):
        response_text = json.dumps({
            'tool_calls': [{'invalid': True}, {'also_invalid': 'yes'}]
        })
        result, _ = self._call(ai_mcp_client_mod, {'response': response_text})
        assert 'error' in result
        assert 'No valid tool calls' in result['error']

    def test_read_timeout_returns_error(self, ai_mcp_client_mod):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        tools = _make_tools(include_text_search=False)
        log = []
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_ollama_with_tools(
                'test', tools, {'ollama_url': 'http://localhost:11434/api/generate'}, log)
        assert 'error' in result
        assert 'timed out' in result['error']

    def test_timeout_exception_returns_error(self, ai_mcp_client_mod):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("connection timeout")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        tools = _make_tools(include_text_search=False)
        log = []
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_ollama_with_tools(
                'test', tools, {'ollama_url': 'http://localhost:11434/api/generate'}, log)
        assert 'error' in result
        assert 'timed out' in result['error']

    def test_generic_exception_returns_ollama_error(self, ai_mcp_client_mod):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("unexpected error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        tools = _make_tools(include_text_search=False)
        log = []
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_ollama_with_tools(
                'test', tools, {'ollama_url': 'http://localhost:11434/api/generate'}, log)
        assert 'error' in result
        assert 'Ollama error' in result['error']

    def test_missing_response_key_returns_error(self, ai_mcp_client_mod):
        result, _ = self._call(ai_mcp_client_mod, {'other_key': 'value'})
        assert 'error' in result
        assert 'Invalid Ollama response' in result['error']


# ---------------------------------------------------------------------------
# TestCallGeminiWithTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallGeminiWithTools:
    """Test _call_gemini_with_tools() - Gemini API mocking."""

    def _make_mock_genai(self):
        """Create a fresh mock google.genai module and install it."""
        mock_genai = MagicMock()
        mock_genai.types = MagicMock()
        mock_genai.types.Tool = MagicMock()
        mock_genai.types.GenerateContentConfig = MagicMock()
        mock_genai.types.ToolConfig = MagicMock()
        mock_genai.types.FunctionCallingConfig = MagicMock()
        return mock_genai

    def _make_response_with_tool_calls(self, tool_calls):
        """Create a mock Gemini response with function_call parts."""
        parts = []
        for tc in tool_calls:
            part = MagicMock()
            fc = MagicMock()
            fc.name = tc['name']
            fc.args = tc.get('arguments', {})
            # Make sure hasattr(fc, 'args') returns True and hasattr(fc, 'arguments') is also accessible
            part.function_call = fc
            parts.append(part)
        candidate = MagicMock()
        candidate.content.parts = parts
        response = MagicMock()
        response.candidates = [candidate]
        return response

    def _call_gemini(self, ai_mcp_client_mod, mock_genai, tools, ai_config, user_msg='test'):
        """Call _call_gemini_with_tools with mock genai injected."""
        # We need to patch sys.modules so `import google.genai as genai` resolves
        google_mock = MagicMock()
        google_mock.genai = mock_genai
        with patch.dict('sys.modules', {
            'google': google_mock,
            'google.genai': mock_genai,
        }):
            return ai_mcp_client_mod._call_gemini_with_tools(
                user_msg, tools, ai_config, [])

    def test_missing_api_key_returns_error(self, ai_mcp_client_mod):
        mock_genai = self._make_mock_genai()
        result = self._call_gemini(
            ai_mcp_client_mod, mock_genai, _make_tools(),
            {'gemini_key': '', 'gemini_model': 'gemini-2.5-pro'})
        assert 'error' in result
        assert 'Valid Gemini API key required' in result['error']

    def test_placeholder_api_key_returns_error(self, ai_mcp_client_mod):
        mock_genai = self._make_mock_genai()
        result = self._call_gemini(
            ai_mcp_client_mod, mock_genai, _make_tools(),
            {'gemini_key': 'YOUR-GEMINI-API-KEY-HERE', 'gemini_model': 'gemini-2.5-pro'})
        assert 'error' in result
        assert 'Valid Gemini API key required' in result['error']

    def test_successful_tool_call_extraction(self, ai_mcp_client_mod):
        mock_genai = self._make_mock_genai()
        response = self._make_response_with_tool_calls([
            {'name': 'search_database', 'arguments': {'genres': ['rock']}}
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_genai.Client.return_value = mock_client

        result = self._call_gemini(
            ai_mcp_client_mod, mock_genai, _make_tools(),
            {'gemini_key': 'real-key-123', 'gemini_model': 'gemini-2.5-pro'},
            user_msg='play rock music')
        assert 'tool_calls' in result
        assert len(result['tool_calls']) == 1
        assert result['tool_calls'][0]['name'] == 'search_database'

    def test_no_tool_calls_returns_error(self, ai_mcp_client_mod):
        mock_genai = self._make_mock_genai()
        response = MagicMock()
        response.candidates = []
        response.text = "I cannot call tools"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_genai.Client.return_value = mock_client

        result = self._call_gemini(
            ai_mcp_client_mod, mock_genai, _make_tools(),
            {'gemini_key': 'real-key-123', 'gemini_model': 'gemini-2.5-pro'})
        assert 'error' in result
        assert 'AI did not call any tools' in result['error']

    def test_exception_returns_gemini_error(self, ai_mcp_client_mod):
        mock_genai = self._make_mock_genai()
        mock_genai.Client.side_effect = RuntimeError("API failure")

        result = self._call_gemini(
            ai_mcp_client_mod, mock_genai, _make_tools(),
            {'gemini_key': 'real-key-123', 'gemini_model': 'gemini-2.5-pro'})
        assert 'error' in result
        assert 'Gemini error' in result['error']

    def test_schema_type_conversion(self, ai_mcp_client_mod):
        """Verify the convert_schema_for_gemini produces uppercase types."""
        mock_genai = self._make_mock_genai()
        response = self._make_response_with_tool_calls([
            {'name': 'test_tool', 'arguments': {}}
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_genai.Client.return_value = mock_client

        # Use a tool with known schema types
        tools = [{
            'name': 'test_tool',
            'description': 'test',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'A name'},
                    'count': {'type': 'number', 'description': 'A count'},
                    'flag': {'type': 'boolean', 'description': 'A flag'},
                    'items': {'type': 'array', 'items': {'type': 'integer'}},
                }
            }
        }]

        self._call_gemini(
            ai_mcp_client_mod, mock_genai, tools,
            {'gemini_key': 'real-key-123', 'gemini_model': 'gemini-2.5-pro'})

        # The Tool() call should have been made with converted schemas
        tool_call = mock_genai.types.Tool.call_args
        # Tool(function_declarations=...) - check keyword arg
        if tool_call[1] and 'function_declarations' in tool_call[1]:
            func_decls = tool_call[1]['function_declarations']
        else:
            # positional: Tool(function_declarations_list)
            func_decls = tool_call[0][0] if tool_call[0] else None
        assert func_decls is not None, "function_declarations not found in Tool() call"
        params = func_decls[0]['parameters']
        assert params['type'] == 'OBJECT'
        assert params['properties']['name']['type'] == 'STRING'
        assert params['properties']['count']['type'] == 'NUMBER'
        assert params['properties']['flag']['type'] == 'BOOLEAN'
        assert params['properties']['items']['type'] == 'ARRAY'
        assert params['properties']['items']['items']['type'] == 'INTEGER'


# ---------------------------------------------------------------------------
# TestCallOpenaiWithTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallOpenaiWithTools:
    """Test _call_openai_with_tools() - OpenAI API mocking."""

    def _make_httpx_client_mock(self, response_data):
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = Mock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        return mock_client

    def test_successful_tool_call_extraction(self, ai_mcp_client_mod):
        import httpx
        response_data = {
            'choices': [{
                'message': {
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': 'search_database',
                            'arguments': json.dumps({'genres': ['rock']})
                        }
                    }]
                }
            }]
        }
        mock_client = self._make_httpx_client_mock(response_data)
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_openai_with_tools(
                'play rock', _make_tools(),
                {'openai_url': 'http://localhost', 'openai_key': 'test', 'openai_model': 'gpt-4'},
                [])
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['name'] == 'search_database'

    def test_no_tool_calls_returns_error(self, ai_mcp_client_mod):
        import httpx
        response_data = {
            'choices': [{
                'message': {
                    'content': 'I found some songs for you'
                }
            }]
        }
        mock_client = self._make_httpx_client_mock(response_data)
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_openai_with_tools(
                'play rock', _make_tools(),
                {'openai_url': 'http://localhost', 'openai_key': 'test', 'openai_model': 'gpt-4'},
                [])
        assert 'error' in result
        assert 'AI did not call any tools' in result['error']

    def test_read_timeout_returns_error(self, ai_mcp_client_mod):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_openai_with_tools(
                'test', _make_tools(),
                {'openai_url': 'http://localhost', 'openai_key': 'test', 'openai_model': 'gpt-4'},
                [])
        assert 'error' in result
        assert 'timed out' in result['error']

    def test_generic_exception_returns_error(self, ai_mcp_client_mod):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("connection failed")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        with patch.object(httpx, 'Client', return_value=mock_client):
            result = ai_mcp_client_mod._call_openai_with_tools(
                'test', _make_tools(),
                {'openai_url': 'http://localhost', 'openai_key': 'test', 'openai_model': 'gpt-4'},
                [])
        assert 'error' in result
        assert 'OpenAI error' in result['error']


# ---------------------------------------------------------------------------
# TestCallMistralWithTools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCallMistralWithTools:
    """Test _call_mistral_with_tools() - Mistral API mocking."""

    def _make_mock_mistral_module(self):
        """Create a mock mistralai module."""
        mock_mod = MagicMock()
        return mock_mod

    def test_missing_api_key_returns_error(self, ai_mcp_client_mod):
        mock_mistral_mod = self._make_mock_mistral_module()
        with patch.dict('sys.modules', {'mistralai': mock_mistral_mod}):
            result = ai_mcp_client_mod._call_mistral_with_tools(
                'test', _make_tools(),
                {'mistral_key': '', 'mistral_model': 'mistral-large-latest'}, [])
        assert 'error' in result
        assert 'Valid Mistral API key required' in result['error']

    def test_placeholder_key_returns_error(self, ai_mcp_client_mod):
        mock_mistral_mod = self._make_mock_mistral_module()
        with patch.dict('sys.modules', {'mistralai': mock_mistral_mod}):
            result = ai_mcp_client_mod._call_mistral_with_tools(
                'test', _make_tools(),
                {'mistral_key': 'YOUR-GEMINI-API-KEY-HERE', 'mistral_model': 'mistral-large-latest'}, [])
        assert 'error' in result
        assert 'Valid Mistral API key required' in result['error']

    def test_successful_tool_call_extraction(self, ai_mcp_client_mod):
        mock_mistral_mod = self._make_mock_mistral_module()
        # Build mock response
        mock_tc = MagicMock()
        mock_tc.function.name = 'search_database'
        mock_tc.function.arguments = json.dumps({'genres': ['jazz']})
        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client_instance = MagicMock()
        mock_client_instance.chat.complete.return_value = mock_response
        mock_mistral_mod.Mistral.return_value = mock_client_instance

        with patch.dict('sys.modules', {'mistralai': mock_mistral_mod}):
            result = ai_mcp_client_mod._call_mistral_with_tools(
                'play jazz', _make_tools(),
                {'mistral_key': 'real-key-abc', 'mistral_model': 'mistral-large-latest'}, [])
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['name'] == 'search_database'

    def test_exception_returns_mistral_error(self, ai_mcp_client_mod):
        mock_mistral_mod = self._make_mock_mistral_module()
        mock_mistral_mod.Mistral.side_effect = RuntimeError("API down")

        with patch.dict('sys.modules', {'mistralai': mock_mistral_mod}):
            result = ai_mcp_client_mod._call_mistral_with_tools(
                'test', _make_tools(),
                {'mistral_key': 'real-key-abc', 'mistral_model': 'mistral-large-latest'}, [])
        assert 'error' in result
        assert 'Mistral error' in result['error']
