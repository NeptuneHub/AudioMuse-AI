"""Unit tests for the new tasks/ai_api*.py modules.

Tests cover AI playlist naming functions including cleaning, API calls,
and provider routing. These tests bypass tasks/__init__.py (which pulls
heavy deps like librosa) by pre-loading the relevant submodules directly.
"""
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock as _MagicMock


# ---------------------------------------------------------------------------
# Bootstrap: load only the AI submodules without running tasks/__init__.py.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)


def _ensure_namespace_pkg(name: str, sub_path: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, sub_path)]
    sys.modules[name] = pkg


def _load_submodule(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO_ROOT, relpath)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stub optional dependencies that aren't installed in the test env.
# ---------------------------------------------------------------------------
def _ensure_httpx_stub():
    if 'httpx' in sys.modules:
        return
    try:
        import httpx  # noqa: F401
        return
    except ImportError:
        pass
    httpx_mod = types.ModuleType('httpx')

    class _ReadTimeout(Exception):
        pass

    class _TimeoutException(Exception):
        pass

    class _Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def post(self, *a, **kw): raise NotImplementedError("stub")

    httpx_mod.ReadTimeout = _ReadTimeout
    httpx_mod.TimeoutException = _TimeoutException
    httpx_mod.Client = _Client
    sys.modules['httpx'] = httpx_mod


def _ensure_google_genai_stub():
    try:
        import google.genai  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    if 'google' not in sys.modules:
        google_mod = types.ModuleType('google')
        google_mod.__path__ = []
        sys.modules['google'] = google_mod
    from unittest.mock import MagicMock as _MM
    from unittest.mock import MagicMock as _MM
    genai_mod = types.ModuleType('google.genai')
    genai_mod.Client = _MM
    genai_types = types.ModuleType('google.genai.types')
    genai_types.Tool = _MM
    genai_types.GenerateContentConfig = _MM
    genai_types.ToolConfig = _MM
    genai_types.FunctionCallingConfig = _MM
    genai_mod.types = genai_types
    sys.modules['google.genai'] = genai_mod
    sys.modules['google.genai.types'] = genai_types


def _ensure_mistralai_stub():
    if 'mistralai' in sys.modules:
        return
    try:
        import mistralai  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    mod = types.ModuleType('mistralai')
    mod.Mistral = _MagicMock  # type: ignore
    sys.modules['mistralai'] = mod


_ensure_namespace_pkg('tasks', 'tasks')
_ensure_httpx_stub()
_ensure_google_genai_stub()
_ensure_mistralai_stub()
for _name, _relpath in (
    ('tasks.ai_prompts',     'tasks/ai_prompts.py'),
    ('tasks.ai_api_openai',  'tasks/ai_api_openai.py'),
    ('tasks.ai_api_ollama',  'tasks/ai_api_ollama.py'),
    ('tasks.ai_api_gemini',  'tasks/ai_api_gemini.py'),
    ('tasks.ai_api_mistral', 'tasks/ai_api_mistral.py'),
    ('tasks.ai_api',         'tasks/ai_api.py'),
):
    _load_submodule(_name, _relpath)


import pytest
from unittest.mock import Mock, MagicMock, patch, call
import requests
import json
from tasks.ai_api import clean_playlist_name, get_ai_playlist_name
from tasks.ai_api_openai import generate_text as get_openai_compatible_playlist_name
from tasks.ai_api_ollama import generate_text as get_ollama_playlist_name
from tasks.ai_api_gemini import generate_text as get_gemini_playlist_name
from tasks.ai_api_mistral import generate_text as get_mistral_playlist_name
from tasks.ai_prompts import creative_prompt_template


class TestCleanPlaylistName:
    """Tests for the clean_playlist_name function"""

    def test_basic_ascii_name(self):
        """Test that basic ASCII names pass through unchanged"""
        name = "Rock Classics"
        assert clean_playlist_name(name) == "Rock Classics"

    def test_removes_special_characters(self):
        """Test removal of special unicode characters"""
        name = "Rock★Classics★"
        result = clean_playlist_name(name)
        assert "★" not in result
        assert result == "RockClassics"

    def test_preserves_allowed_punctuation(self):
        """Test that allowed punctuation is preserved"""
        name = "Rock & Roll - 80's Hits! (Best)"
        result = clean_playlist_name(name)
        assert result == "Rock & Roll - 80's Hits! (Best)"

    def test_removes_trailing_number_parentheses(self):
        """Test removal of trailing disambiguation numbers"""
        name = "My Playlist (2)"
        result = clean_playlist_name(name)
        assert result == "My Playlist"

    def test_handles_non_string_input(self):
        """Test returns empty string for non-string input"""
        assert clean_playlist_name(None) == ""
        assert clean_playlist_name(123) == ""
        assert clean_playlist_name([]) == ""

    def test_normalizes_unicode(self):
        """Test Unicode normalization (NFKC)"""
        name = "Café"  # Different unicode representations
        result = clean_playlist_name(name)
        assert isinstance(result, str)
        assert "Caf" in result

    def test_collapses_multiple_spaces(self):
        """Test that multiple spaces are collapsed to single space"""
        name = "Rock    Classics"
        result = clean_playlist_name(name)
        assert result == "Rock Classics"

    def test_strips_leading_trailing_whitespace(self):
        """Test whitespace stripping"""
        name = "  Rock Classics  "
        result = clean_playlist_name(name)
        assert result == "Rock Classics"

    def test_fixes_text_encoding(self):
        """Test ftfy text encoding fixes"""
        # ftfy should fix common encoding issues
        name = "Rock Classics"
        result = clean_playlist_name(name)
        assert isinstance(result, str)


class TestGetOpenAICompatiblePlaylistName:
    """Tests for OpenAI-compatible API function"""

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_openai_format_success(self, mock_sleep, mock_post):
        """Test successful OpenAI format API call"""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        # Simulate SSE streaming chunks
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Sunset"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" Vibes"}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop"}]}\n'
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4",
            full_prompt="Create a playlist name",
            api_key="test-key"
        )

        assert result == "Sunset Vibes"
        assert mock_sleep.called  # Should delay for rate limiting

    @patch('tasks.ai_api_openai.requests.post')
    def test_ollama_format_success(self, mock_post):
        """Test successful Ollama format API call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        # Simulate Ollama streaming chunks
        chunks = [
            b'{"response":"Morning","done":false}\n',
            b'{"response":" Calm","done":false}\n',
            b'{"response":"","done":true}\n'
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="deepseek-r1:1.5b",
            full_prompt="Create a playlist name",
            api_key="no-key-needed"
        )

        assert result == "Morning Calm"

    @patch('tasks.ai_api_openai.requests.post')
    def test_handles_think_tags(self, mock_post):
        """Test extraction of text after think tags"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [
            b'{"response":"<think>reasoning here</think>Final Name","done":true}\n'
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="model",
            full_prompt="test",
            api_key="no-key-needed"
        )

        assert result == "Final Name"
        assert "<think>" not in result

    @patch('tasks.ai_api_openai.requests.post')
    def test_handles_api_error(self, mock_post):
        """Test handling of API request errors"""
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        result = get_openai_compatible_playlist_name(
            server_url="http://invalid",
            model_name="model",
            full_prompt="test",
            api_key="key"
        )

        assert "Error" in result
        assert "unavailable" in result

    @patch('tasks.ai_api_openai.requests.post')
    def test_handles_invalid_json(self, mock_post):
        """Test handling of malformed JSON responses"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [
            b'invalid json\n',
            b'{"response":"Valid","done":true}\n'
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="model",
            full_prompt="test",
            api_key="no-key-needed"
        )

        # Should still extract the valid chunk
        assert result == "Valid"

    @patch('tasks.ai_api_openai.requests.post')
    def test_openrouter_headers(self, mock_post):
        """Test OpenRouter-specific headers are added"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Test"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]
        mock_post.return_value = mock_response

        get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4",
            full_prompt="test",
            api_key="test-key"
        )

        # Check headers include OpenRouter-specific ones
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_combined_content_and_finish_reason_chunk(self, mock_sleep, mock_post):
        """Regression test for issue #467.

        OpenRouter (and some providers behind it, notably Anthropic via Vertex)
        can emit a single SSE chunk that contains both ``delta.content`` AND
        ``finish_reason: "stop"``. The parser must extract the content from
        such a chunk before terminating the loop. Short outputs (5-12 token
        playlist titles) are particularly vulnerable because the entire
        response can fit in one or two such combined chunks.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Quiet Storm"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4o-mini",
            full_prompt="prompt",
            api_key="test-key",
        )

        assert result == "Quiet Storm"
        # Success path must not trigger the empty-content retry/backoff.
        # The pre-call rate-limit delay calls time.sleep once with skip_delay=False;
        # ensure no additional retry-backoff sleeps happened.
        assert mock_sleep.call_count == 1

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_content_split_across_chunks_with_final_combined_chunk(self, mock_sleep, mock_post):
        """Regression test for issue #467 (multi-chunk variant).

        Some content arrives in early chunks and the LAST content fragment is
        bundled with finish_reason='stop' in a single chunk. All fragments
        (including the bundled one) must be captured.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Quiet "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"Storm"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4o-mini",
            full_prompt="prompt",
            api_key="test-key",
        )

        assert result == "Quiet Storm"
        # Success path must not trigger the empty-content retry/backoff.
        # Only the pre-call rate-limit delay (one sleep) is expected.
        assert mock_sleep.call_count == 1

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_authenticated_ollama_url_uses_ollama_format(self, mock_sleep, mock_post):
        """Detection must use the URL, not the api_key, so a real bearer token
        passed to an Ollama deployment (e.g. Ollama behind a reverse proxy or
        LiteLLM with auth) still uses Ollama's request/response format.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        # Ollama-format response chunks (response/done keys, not delta/text).
        mock_response.iter_lines.return_value = [
            b'{"response":"Quiet","done":false}\n',
            b'{"response":" Storm","done":false}\n',
            b'{"response":"","done":true}\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://ollama-proxy.example.com:11434/api/generate",
            model_name="llama3.1:8b",
            full_prompt="prompt",
            api_key="real-bearer-token",
        )

        assert result == "Quiet Storm"
        # And the request body must use Ollama's `prompt` field, not OpenAI's `messages`.
        sent_body = json.loads(mock_post.call_args[1]['data'])
        assert 'prompt' in sent_body
        assert 'messages' not in sent_body

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_openai_format_detected_from_url_when_global_is_ollama(self, mock_sleep, mock_post):
        """Regression test for issue #467 — provider-mismatch root cause.

        When the global AI_MODEL_PROVIDER is set to OLLAMA (e.g. user's default
        for cron jobs) but a clustering call hits OpenRouter with a real API
        key, the parser must still treat the response as OpenAI chat-completion
        format (delta.content / choice.text), not Ollama format (response/done).

        Before the fix, the function read AI_MODEL_PROVIDER from the global
        config and used it to decide both payload AND parser format, dropping
        all chunks because they had `choices` instead of `response`.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        # Real OpenRouter-via-Bedrock chunk shape: choice.text (no delta).
        mock_response.iter_lines.return_value = [
            b':' + b' OPENROUTER PROCESSING\n',
            b'data: {"choices":[{"index":0,"finish_reason":null,"text":"Late"}]}\n',
            b'data: {"choices":[{"index":0,"finish_reason":null,"text":" Night Blues"}]}\n',
            b'data: {"choices":[{"index":0,"finish_reason":"stop","text":""}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="anthropic/claude-sonnet-4.6",
            full_prompt="prompt",
            api_key="sk-or-v1-test",
        )

        assert result == "Late Night Blues"
        # And the request body must use OpenAI's `messages` field, not Ollama's
        # `prompt` field — i.e. the format detection also drove the payload.
        sent_body = json.loads(mock_post.call_args[1]['data'])
        assert 'messages' in sent_body
        assert 'prompt' not in sent_body

    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_rate_limit_retry_with_exponential_backoff(self, mock_sleep, mock_post):
        """Test that rate limit errors (429) retry with exponential backoff"""
        # First call: 429 rate limit
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_429)

        # Second call: Success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_429, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Success"
        # Should have delayed with base_delay * (2 ** 0) = 5 seconds for first retry
        assert mock_sleep.call_count >= 2  # Initial delay + retry delay
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 5 in sleep_calls  # Exponential backoff for attempt 0

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_aggressive_fallback_on_unsupported_parameter(self, mock_sleep, mock_post, mock_env):
        """Test aggressive fallback removes temperature and switches to max_completion_tokens"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with unsupported parameter
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                'type': 'invalid_request_error',
                'param': 'max_tokens',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400)

        # Second call: Success (content comes before finish_reason)
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Fallback Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Fallback Success"
        # Should be exactly 2 calls (initial + fallback retry)
        assert mock_post.call_count == 2
        
        # Check second call has modified payload
        second_call_data = json.loads(mock_post.call_args_list[1][1]['data'])
        assert 'temperature' not in second_call_data
        assert 'max_tokens' not in second_call_data
        assert second_call_data.get('max_completion_tokens') == 8000

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_ultra_minimal_fallback_after_aggressive_fails(self, mock_sleep, mock_post, mock_env):
        """Test ultra-minimal fallback removes max_completion_tokens if aggressive fails"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with unsupported parameter
        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_value'
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_1)

        # Second call: Still 400 with max_completion_tokens
        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_2)

        # Third call: Success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Ultra Minimal"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Ultra Minimal"
        # Should be exactly 3 calls (initial + aggressive fallback + ultra-minimal fallback)
        assert mock_post.call_count == 3
        
        # Check third call has minimal payload (no token limits, no temperature)
        third_call_data = json.loads(mock_post.call_args_list[2][1]['data'])
        assert 'temperature' not in third_call_data
        assert 'max_tokens' not in third_call_data
        assert 'max_completion_tokens' not in third_call_data

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_rate_limit_then_parameter_error(self, mock_sleep, mock_post, mock_env):
        """Test rate limit retry followed by parameter error fallback"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 429 rate limit
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_429)

        # Second call: 400 unsupported parameter
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400)

        # Third call: Success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Combined Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_429, mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Combined Success"
        # Should be exactly 3 calls (initial with 429, retry with 400, fallback success)
        assert mock_post.call_count == 3
        
        # Check that sleep was called for rate limit (exponential backoff)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0][0] >= 5]
        assert len(sleep_calls) >= 1  # At least one sleep for rate limit

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    @patch('tasks.ai_api_openai.time.sleep')
    def test_parameter_fallbacks_dont_consume_retry_budget(self, mock_sleep, mock_post, mock_env):
        """Test that parameter fallbacks use continue and don't increment attempt counter"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # We'll simulate: 400 (unsupported) -> 400 (still unsupported) -> timeout -> success
        # This tests that fallbacks don't consume the retry budget
        
        # First call: 400 with unsupported
        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_1)

        # Second call: 400 still unsupported (max_completion_tokens)
        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_2)

        # Third call: Success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Final Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Final Success"
        # Should be exactly 3 calls total
        assert mock_post.call_count == 3

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    def test_existing_max_tokens_fallback_still_works(self, mock_post, mock_env):
        """Test that max_tokens parameter errors with error code 'unsupported_parameter' are handled"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with max_tokens not supported (using proper error code)
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                'type': 'invalid_request_error',
                'param': 'max_tokens',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400)

        # Second call: Success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Max Tokens Fallback"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        ]

        mock_post.side_effect = [mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key"
        )

        assert result == "Max Tokens Fallback"
        # Should have switched to max_completion_tokens
        second_call_data = json.loads(mock_post.call_args_list[1][1]['data'])
        assert 'max_tokens' not in second_call_data
        assert second_call_data.get('max_completion_tokens') == 8000

    @patch('tasks.ai_api_openai.os.environ.get')
    @patch('tasks.ai_api_openai.requests.post')
    def test_ultra_minimal_fallback_requires_proper_error_code(self, mock_post, mock_env):
        """Test that ultra-minimal fallback only triggers with error codes 'unsupported_parameter' or 'unsupported_value'"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with proper error code (triggers aggressive fallback)
        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter'
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_1)

        # Second call: 400 but WITHOUT proper error code (should NOT trigger ultra-minimal)
        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': 'Invalid parameter: max_completion_tokens',
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'invalid_parameter'  # Different error code
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.text = 'Invalid parameter'
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_2)

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key"
        )

        # Should return error, not trigger ultra-minimal fallback
        assert "Error" in result
        # Should only have made 2 calls (initial + aggressive fallback, then error)
        assert mock_post.call_count == 2


class TestGetOllamaPlaylistName:
    """Tests for Ollama-specific wrapper function"""

    @patch('tasks.ai_api_openai.generate_text')
    def test_calls_openai_compatible_with_correct_params(self, mock_func):
        """Test that Ollama wrapper calls underlying function correctly"""
        mock_func.return_value = "Test Playlist"

        result = get_ollama_playlist_name(
            ollama_url="http://localhost:11434/api/generate",
            model_name="deepseek-r1:1.5b",
            full_prompt="test prompt"
        )

        mock_func.assert_called_once_with(
            "http://localhost:11434/api/generate",
            "deepseek-r1:1.5b",
            "test prompt",
            api_key="no-key-needed",
            skip_delay=False
        )
        assert result == "Test Playlist"


class TestGetGeminiPlaylistName:
    """Tests for Google Gemini API function"""

    @patch('google.genai.Client')
    @patch('tasks.ai_api_gemini.time.sleep')
    def test_successful_gemini_call(self, mock_sleep, mock_client_class):
        """Test successful Gemini API call"""
        # Mock response structure for new google-genai API
        mock_response = Mock()
        mock_response.text = "Chill Vibes"

        mock_models = Mock()
        mock_models.generate_content.return_value = mock_response

        mock_client = Mock()
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        result = get_gemini_playlist_name(
            api_key="valid-key",
            model_name="gemini-2.5-pro",
            full_prompt="Create a name"
        )

        assert result == "Chill Vibes"
        mock_client_class.assert_called_once_with(api_key="valid-key")
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        """Test that empty API key returns error"""
        result = get_gemini_playlist_name(
            api_key="",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        """Test that placeholder API key returns error"""
        result = get_gemini_playlist_name(
            api_key="YOUR-GEMINI-API-KEY-HERE",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result

    @patch('google.genai.Client')
    @patch('tasks.ai_api_gemini.time.sleep')
    def test_handles_gemini_api_error(self, mock_sleep, mock_client_class):
        """Test handling of Gemini API errors"""
        mock_models = Mock()
        mock_models.generate_content.side_effect = Exception("API Error")

        mock_client = Mock()
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        result = get_gemini_playlist_name(
            api_key="valid-key",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result
        assert "unavailable" in result


class TestGetMistralPlaylistName:
    """Tests for Mistral API function"""

    @patch('mistralai.Mistral')
    @patch('tasks.ai_api_mistral.time.sleep')
    def test_successful_mistral_call(self, mock_sleep, mock_mistral_class):
        """Test successful Mistral API call"""
        # Mock response structure
        mock_message = Mock()
        mock_message.content = "Electronic Dreams"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_chat = Mock()
        mock_chat.complete.return_value = mock_response

        mock_client = Mock()
        mock_client.chat = mock_chat
        mock_mistral_class.return_value = mock_client

        result = get_mistral_playlist_name(
            api_key="valid-key",
            model_name="ministral-3b-latest",
            full_prompt="Create a name"
        )

        assert result == "Electronic Dreams"
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        """Test that empty API key returns error"""
        result = get_mistral_playlist_name(
            api_key="",
            model_name="ministral-3b-latest",
            full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        """Test that placeholder API key returns error"""
        result = get_mistral_playlist_name(
            api_key="YOUR-MISTRAL-API-KEY-HERE",
            model_name="ministral-3b-latest",
            full_prompt="test"
        )

        assert "Error" in result


class TestGetAIPlaylistName:
    """Tests for the main AI playlist name orchestration function.

    The new dispatcher signature is:
        get_ai_playlist_name(prompt_template, song_list, other_feature_scores_dict, ai_config)

    Provider routing is driven by ``ai_config['provider']`` and dispatched
    through ``tasks.ai_api.generate_text``. Tests here mock that single entry
    point — provider-specific transports are covered by their own test classes.
    """

    @staticmethod
    def _ai_config(provider, **extra):
        cfg = {"provider": provider}
        cfg.update(extra)
        return cfg

    @patch('tasks.ai_api.generate_text')
    def test_routes_to_ollama(self, mock_generate):
        mock_generate.return_value = "Test Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Song 1", "author": "Artist 1"}],
            {"energy": 0.8},
            self._ai_config("OLLAMA",
                            ollama_url="http://localhost:11434/api/generate",
                            ollama_model="deepseek-r1:1.5b"),
        )

        assert result == "Test Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai_api.generate_text')
    def test_routes_to_gemini(self, mock_generate):
        mock_generate.return_value = "Gemini Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Song 1", "author": "Artist 1"}],
            {},
            self._ai_config("GEMINI", gemini_key="valid-key", gemini_model="gemini-2.5-pro"),
        )

        assert result == "Gemini Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai_api.generate_text')
    def test_routes_to_mistral(self, mock_generate):
        mock_generate.return_value = "Mistral Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Symphony", "author": "Beethoven"}],
            {},
            self._ai_config("MISTRAL", mistral_key="valid-key", mistral_model="ministral-3b-latest"),
        )

        assert result == "Mistral Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai_api.generate_text')
    def test_routes_to_openai(self, mock_generate):
        mock_generate.return_value = "OpenAI Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Track", "author": "Artist"}],
            {},
            self._ai_config("OPENAI",
                            openai_url="https://api.openai.com/v1/chat/completions",
                            openai_model="gpt-4",
                            openai_key="test-key"),
        )

        assert result == "OpenAI Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai_api.generate_text')
    def test_handles_none_provider(self, mock_generate):
        """NONE provider is signalled by generate_text returning the skip sentinel."""
        mock_generate.return_value = "AI Naming Skipped"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [],
            {},
            self._ai_config("NONE"),
        )

        assert result == "AI Naming Skipped"

    @patch('tasks.ai_api.clean_playlist_name')
    @patch('tasks.ai_api.generate_text')
    def test_applies_length_constraints(self, mock_generate, mock_clean):
        """Names shorter than MIN_LENGTH (5) cause an error after retries."""
        mock_generate.return_value = "Test"
        mock_clean.return_value = "Test"  # 4 chars — too short

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Test", "author": "Artist"}],
            {},
            self._ai_config("OLLAMA",
                            ollama_url="http://localhost:11434/api/generate",
                            ollama_model="model"),
        )

        assert "Error" in result
        assert "outside" in result

    @patch('tasks.ai_api.clean_playlist_name')
    @patch('tasks.ai_api.generate_text')
    def test_cleans_playlist_name(self, mock_generate, mock_clean):
        mock_generate.return_value = "Rock & Roll - Best Hits!"
        mock_clean.return_value = "Rock & Roll - Best Hits!"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Test", "author": "Artist"}],
            {},
            self._ai_config("OLLAMA",
                            ollama_url="http://localhost:11434/api/generate",
                            ollama_model="model"),
        )

        mock_clean.assert_called_once()
        assert result == "Rock & Roll - Best Hits!"

    @patch('tasks.ai_api.generate_text')
    def test_formats_song_list_in_prompt(self, mock_generate):
        """The {song_list_sample} placeholder must contain title and author lines."""
        mock_generate.return_value = "Test Playlist Name"
        song_list = [
            {"title": "Song One", "author": "Artist A"},
            {"title": "Song Two", "author": "Artist B"},
        ]

        get_ai_playlist_name(
            creative_prompt_template,
            song_list,
            {},
            self._ai_config("OLLAMA",
                            ollama_url="http://localhost:11434/api/generate",
                            ollama_model="model"),
        )

        prompt = mock_generate.call_args[0][0]  # first positional arg
        assert "Song One" in prompt
        assert "Artist A" in prompt
        assert "Song Two" in prompt
        assert "Artist B" in prompt

    def test_prompt_includes_length_requirement(self):
        """Ensure the prompt specifies the 5-40 character length requirement"""
        assert "The title MUST be within the range of 5 to 40 characters long." in creative_prompt_template
