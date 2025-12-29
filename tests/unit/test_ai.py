"""Unit tests for ai.py

Tests cover AI playlist naming functions including cleaning, API calls,
and provider routing.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import requests
import json
from ai import (
    clean_playlist_name,
    get_openai_compatible_playlist_name,
    get_ollama_playlist_name,
    get_gemini_playlist_name,
    get_mistral_playlist_name,
    get_ai_playlist_name,
    creative_prompt_template
)


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

    @patch('ai.requests.post')
    @patch('ai.time.sleep')
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

    @patch('ai.requests.post')
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

    @patch('ai.requests.post')
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

    @patch('ai.requests.post')
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

    @patch('ai.requests.post')
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

    @patch('ai.requests.post')
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

    @patch('ai.requests.post')
    @patch('ai.time.sleep')
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

    @patch('ai.os.environ.get')
    @patch('ai.requests.post')
    @patch('ai.time.sleep')
    def test_aggressive_fallback_on_unsupported_parameter(self, mock_sleep, mock_post, mock_env):
        """Test aggressive fallback removes temperature and switches to max_completion_tokens"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with unsupported parameter
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': 'Extra inputs are not permitted. The following unsupported parameters were provided: temperature, max_tokens'
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

    @patch('ai.os.environ.get')
    @patch('ai.requests.post')
    @patch('ai.time.sleep')
    def test_ultra_minimal_fallback_after_aggressive_fails(self, mock_sleep, mock_post, mock_env):
        """Test ultra-minimal fallback removes max_completion_tokens if aggressive fails"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with unsupported parameter
        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': 'Extra inputs are not permitted. The following unsupported parameters were provided: temperature'
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_1)

        # Second call: Still 400 with max_completion_tokens
        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': 'Extra inputs are not permitted. The following unsupported parameters: max_completion_tokens'
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

    @patch('ai.os.environ.get')
    @patch('ai.requests.post')
    @patch('ai.time.sleep')
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
                'message': 'Extra inputs are not permitted. unsupported: temperature'
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

    @patch('ai.os.environ.get')
    @patch('ai.requests.post')
    @patch('ai.time.sleep')
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
                'message': 'unsupported parameter: temperature'
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_400_1)

        # Second call: 400 still unsupported (max_completion_tokens)
        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': 'unsupported: max_completion_tokens'
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

    @patch('ai.os.environ.get')
    @patch('ai.requests.post')
    def test_existing_max_tokens_fallback_still_works(self, mock_post, mock_env):
        """Test that existing max_tokens fallback logic is preserved"""
        # Disable initial delay for cleaner testing
        mock_env.return_value = "0"
        
        # First call: 400 with max_tokens not supported
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': 'max_tokens is not supported for this model'
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


class TestGetOllamaPlaylistName:
    """Tests for Ollama-specific wrapper function"""

    @patch('ai.get_openai_compatible_playlist_name')
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

    @patch('ai.genai.GenerativeModel')
    @patch('ai.genai.configure')
    @patch('ai.time.sleep')
    def test_successful_gemini_call(self, mock_sleep, mock_configure, mock_model_class):
        """Test successful Gemini API call"""
        # Mock response structure
        mock_part = Mock()
        mock_part.text = "Chill Vibes"

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        result = get_gemini_playlist_name(
            gemini_api_key="valid-key",
            model_name="gemini-2.5-pro",
            full_prompt="Create a name"
        )

        assert result == "Chill Vibes"
        mock_configure.assert_called_once_with(api_key="valid-key")
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        """Test that empty API key returns error"""
        result = get_gemini_playlist_name(
            gemini_api_key="",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        """Test that placeholder API key returns error"""
        result = get_gemini_playlist_name(
            gemini_api_key="YOUR-GEMINI-API-KEY-HERE",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result

    @patch('ai.genai.GenerativeModel')
    @patch('ai.genai.configure')
    @patch('ai.time.sleep')
    def test_handles_gemini_api_error(self, mock_sleep, mock_configure, mock_model_class):
        """Test handling of Gemini API errors"""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        result = get_gemini_playlist_name(
            gemini_api_key="valid-key",
            model_name="gemini-2.5-pro",
            full_prompt="test"
        )

        assert "Error" in result
        assert "unavailable" in result


class TestGetMistralPlaylistName:
    """Tests for Mistral API function"""

    @patch('ai.Mistral')
    @patch('ai.time.sleep')
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
            mistral_api_key="valid-key",
            model_name="ministral-3b-latest",
            full_prompt="Create a name"
        )

        assert result == "Electronic Dreams"
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        """Test that empty API key returns error"""
        result = get_mistral_playlist_name(
            mistral_api_key="",
            model_name="ministral-3b-latest",
            full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        """Test that placeholder API key returns error"""
        result = get_mistral_playlist_name(
            mistral_api_key="YOUR-MISTRAL-API-KEY-HERE",
            model_name="ministral-3b-latest",
            full_prompt="test"
        )

        assert "Error" in result


class TestGetAIPlaylistName:
    """Tests for the main AI playlist name orchestration function"""

    @patch('ai.get_ollama_playlist_name')
    def test_routes_to_ollama(self, mock_ollama):
        """Test provider routing to Ollama"""
        mock_ollama.return_value = "Test Playlist"

        result = get_ai_playlist_name(
            provider="OLLAMA",
            ollama_url="http://localhost:11434/api/generate",
            ollama_model_name="deepseek-r1:1.5b",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="rock",
            feature2="energetic",
            feature3="upbeat",
            song_list=[{"title": "Song 1", "author": "Artist 1"}],
            other_feature_scores_dict={"energy": 0.8}
        )

        assert result == "Test Playlist"
        mock_ollama.assert_called_once()

    @patch('ai.get_gemini_playlist_name')
    def test_routes_to_gemini(self, mock_gemini):
        """Test provider routing to Gemini"""
        mock_gemini.return_value = "Gemini Playlist"

        result = get_ai_playlist_name(
            provider="GEMINI",
            ollama_url="",
            ollama_model_name="",
            gemini_api_key="valid-key",
            gemini_model_name="gemini-2.5-pro",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="jazz",
            feature2="smooth",
            feature3="relaxed",
            song_list=[{"title": "Song 1", "author": "Artist 1"}],
            other_feature_scores_dict={}
        )

        assert result == "Gemini Playlist"
        mock_gemini.assert_called_once()

    @patch('ai.get_mistral_playlist_name')
    def test_routes_to_mistral(self, mock_mistral):
        """Test provider routing to Mistral"""
        mock_mistral.return_value = "Mistral Playlist"

        result = get_ai_playlist_name(
            provider="MISTRAL",
            ollama_url="",
            ollama_model_name="",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="valid-key",
            mistral_model_name="ministral-3b-latest",
            prompt_template=creative_prompt_template,
            feature1="classical",
            feature2="peaceful",
            feature3="calm",
            song_list=[{"title": "Symphony", "author": "Beethoven"}],
            other_feature_scores_dict={}
        )

        assert result == "Mistral Playlist"
        mock_mistral.assert_called_once()

    @patch('ai.get_openai_compatible_playlist_name')
    def test_routes_to_openai(self, mock_openai):
        """Test provider routing to OpenAI"""
        mock_openai.return_value = "OpenAI Playlist"

        result = get_ai_playlist_name(
            provider="OPENAI",
            ollama_url="",
            ollama_model_name="",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="hip-hop",
            feature2="energetic",
            feature3="modern",
            song_list=[{"title": "Track", "author": "Artist"}],
            other_feature_scores_dict={},
            openai_server_url="https://api.openai.com/v1/chat/completions",
            openai_model_name="gpt-4",
            openai_api_key="test-key"
        )

        assert result == "OpenAI Playlist"
        mock_openai.assert_called_once()

    def test_handles_none_provider(self):
        """Test handling of NONE provider"""
        result = get_ai_playlist_name(
            provider="NONE",
            ollama_url="",
            ollama_model_name="",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="",
            feature2="",
            feature3="",
            song_list=[],
            other_feature_scores_dict={}
        )

        assert result == "AI Naming Skipped"

    @patch('ai.clean_playlist_name')
    @patch('ai.get_ollama_playlist_name')
    def test_applies_length_constraints(self, mock_ollama, mock_clean):
        """Test that length constraints are enforced"""
        # Name too short (MIN_LENGTH is 5, so use 4 chars)
        mock_ollama.return_value = "Test"
        mock_clean.return_value = "Test"

        result = get_ai_playlist_name(
            provider="OLLAMA",
            ollama_url="http://localhost:11434/api/generate",
            ollama_model_name="model",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="",
            feature2="",
            feature3="",
            song_list=[{"title": "Test", "author": "Artist"}],
            other_feature_scores_dict={}
        )

        assert "Error" in result
        assert "outside" in result

    @patch('ai.clean_playlist_name')
    @patch('ai.get_ollama_playlist_name')
    def test_cleans_playlist_name(self, mock_ollama, mock_clean):
        """Test that playlist names are cleaned"""
        mock_ollama.return_value = "Rock & Roll - Best Hits!"
        mock_clean.return_value = "Rock & Roll - Best Hits!"

        result = get_ai_playlist_name(
            provider="OLLAMA",
            ollama_url="http://localhost:11434/api/generate",
            ollama_model_name="model",
            gemini_api_key="",
            gemini_model_name="",
            mistral_api_key="",
            mistral_model_name="",
            prompt_template=creative_prompt_template,
            feature1="",
            feature2="",
            feature3="",
            song_list=[{"title": "Test", "author": "Artist"}],
            other_feature_scores_dict={}
        )

        mock_clean.assert_called_once()
        assert result == "Rock & Roll - Best Hits!"

    def test_formats_song_list_in_prompt(self):
        """Test that song list is properly formatted in prompt"""
        song_list = [
            {"title": "Song One", "author": "Artist A"},
            {"title": "Song Two", "author": "Artist B"}
        ]

        # We can't easily test the full prompt without mocking,
        # but we can test that the function constructs it
        with patch('ai.get_ollama_playlist_name') as mock_ollama:
            mock_ollama.return_value = "Test Playlist Name"

            get_ai_playlist_name(
                provider="OLLAMA",
                ollama_url="http://localhost:11434/api/generate",
                ollama_model_name="model",
                gemini_api_key="",
                gemini_model_name="",
                mistral_api_key="",
                mistral_model_name="",
                prompt_template=creative_prompt_template,
                feature1="",
                feature2="",
                feature3="",
                song_list=song_list,
                other_feature_scores_dict={}
            )

            # Check that the prompt was passed to Ollama
            call_args = mock_ollama.call_args
            prompt = call_args[0][2]  # third argument is full_prompt
            assert "Song One" in prompt
            assert "Artist A" in prompt
            assert "Song Two" in prompt
            assert "Artist B" in prompt
