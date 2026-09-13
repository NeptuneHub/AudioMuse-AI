# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the clustering playlist naming styles.

Covers the restored full-title naming style and how it coexists with the default
word-plus-genre style: the default instructions are pinned to the exact version 2
prompt that shipped unchanged from v1.1.1 through v2.1.5, the song sample is always appended by
code so an edited prompt can never change the data sent, and the clustering naming
entry point only takes the new path when the title style is selected.

Main Features:
* The default title instructions plus the playlist block hash to the version 2
  template, and render byte-identical to the old song-list format.
* get_ai_playlist_title keeps the old contract: 5-40 character cleaned titles, length
  feedback retries, no temperature override, and no retry on a provider error.
* In the default style the clustering naming path never calls the title function, and
  in the title style it never queries score rows nor calls the word-plus-genre path.
"""

import hashlib
from unittest.mock import patch

import pytest

import config
from tasks import clustering_helper
from tasks.ai.api import get_ai_playlist_title
from tasks.ai.prompts import (
    NAMING_MODES,
    TITLE_PROMPT_PLAYLIST_HEADER,
    build_title_naming_prompt,
    normalize_naming_mode,
    title_prompt_song_block,
)

LEGACY_TITLE_TEMPLATE_SHA256 = (
    '421d5f79e498833b11568af9b523f29d6a51a0ab69b4405029d3c0131f5029dc'
)

SONGS = [
    ('1', 'Vienna', 'Billy Joel'),
    ('2', 'Skinny Love', 'Bon Iver'),
]


def ai_config():
    return {
        'provider': 'OLLAMA',
        'ollama_url': 'http://localhost:11434/api/generate',
        'ollama_model': 'test-model',
        'openai_url': '', 'openai_model': '', 'openai_key': '',
        'gemini_key': '', 'gemini_model': '',
        'mistral_key': '', 'mistral_model': '',
    }


def legacy_template():
    return config._AI_NAMING_TITLE_PROMPT_DEFAULT + 'This is the playlist:\n{song_list_sample}\n\n'


class TestDefaultTitlePrompt:
    def test_default_is_the_exact_version_2_prompt(self):
        digest = hashlib.sha256(legacy_template().encode('utf-8')).hexdigest()
        assert digest == LEGACY_TITLE_TEMPLATE_SHA256

    def test_default_render_is_byte_identical_to_the_legacy_render(self):
        legacy = legacy_template().format(
            song_list_sample='- Vienna by Billy Joel\n- Skinny Love by Bon Iver'
        )
        assert build_title_naming_prompt(
            config._AI_NAMING_TITLE_PROMPT_DEFAULT, SONGS, 25
        ) == legacy

    def test_the_bad_example_holds_real_unicode_letters_not_escape_text(self):
        default = config._AI_NAMING_TITLE_PROMPT_DEFAULT
        assert '\\U0001' not in default
        assert '\U0001d5dd' in default

    def test_the_editable_default_carries_no_song_list_placeholder(self):
        default = config._AI_NAMING_TITLE_PROMPT_DEFAULT
        assert '{song_list_sample}' not in default
        assert 'This is the playlist:' not in default

    def test_the_shipped_naming_style_is_word_plus_genre(self):
        assert config.AI_NAMING_PROMPT_MODE == 'concept'

    def test_the_active_title_prompt_defaults_to_the_pristine_copy(self):
        assert config.AI_NAMING_TITLE_PROMPT == config._AI_NAMING_TITLE_PROMPT_DEFAULT


class TestTitlePromptBuilder:
    def test_the_song_sample_is_capped(self):
        block = title_prompt_song_block(SONGS, 1)
        assert block == TITLE_PROMPT_PLAYLIST_HEADER + '- Vienna by Billy Joel\n\n'

    def test_a_cap_below_one_still_sends_one_song(self):
        assert '- Vienna by Billy Joel' in title_prompt_song_block(SONGS, 0)

    def test_missing_title_and_artist_use_the_legacy_placeholders(self):
        block = title_prompt_song_block([('1', None, '')], 5)
        assert '- Unknown Title by Unknown Artist' in block

    def test_instructions_without_a_trailing_newline_are_kept_apart_from_the_songs(self):
        prompt = build_title_naming_prompt('Name it.', SONGS, 5)
        assert prompt.startswith('Name it.\n\nThis is the playlist:\n')

    def test_braces_in_edited_instructions_are_sent_literally(self):
        prompt = build_title_naming_prompt('Use {genre} and {{x}} literally.', SONGS, 5)
        assert prompt.startswith('Use {genre} and {{x}} literally.')

    def test_a_typed_placeholder_neither_injects_nor_duplicates_the_songs(self):
        prompt = build_title_naming_prompt('Title please {song_list_sample}', SONGS, 5)
        assert prompt.startswith('Title please {song_list_sample}\n\n')
        assert prompt.count('This is the playlist:') == 1
        assert prompt.count('- Vienna by Billy Joel') == 1

    def test_edited_instructions_cannot_remove_the_song_sample(self):
        prompt = build_title_naming_prompt('Ignore the songs.', SONGS, 5)
        assert prompt.endswith('- Vienna by Billy Joel\n- Skinny Love by Bon Iver\n\n')


class TestNormalizeMode:
    @pytest.mark.parametrize('raw, expected', [
        ('title', 'title'), (' TITLE ', 'title'), ('concept', 'concept'),
        ('', 'concept'), (None, 'concept'), ('nonsense', 'concept'),
    ])
    def test_values(self, raw, expected):
        assert normalize_naming_mode(raw) == expected

    def test_two_styles_exist(self):
        assert NAMING_MODES == ('concept', 'title')


class TestGetAiPlaylistTitle:
    @patch('tasks.ai.api.generate_text')
    def test_returns_the_cleaned_title(self, mock_generate):
        mock_generate.return_value = '  Raindrops on the Windshield \n'
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) == (
            'Raindrops on the Windshield'
        )

    @patch('tasks.ai.api.generate_text')
    def test_sends_the_prompt_without_a_temperature_override(self, mock_generate):
        mock_generate.return_value = 'Velvet Morning Light'
        get_ai_playlist_title('Name it.', SONGS, ai_config())
        args, kwargs = mock_generate.call_args
        assert args[0] == build_title_naming_prompt('Name it.', SONGS, config.MAX_SONGS_IN_AI_PROMPT)
        assert kwargs == {}

    @patch('tasks.ai.api.generate_text')
    def test_a_too_long_title_is_retried_with_length_feedback(self, mock_generate):
        mock_generate.side_effect = ['X' * 60, 'Velvet Morning Light']
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) == 'Velvet Morning Light'
        second_prompt = mock_generate.call_args_list[1][0][0]
        assert 'FEEDBACK: The previous title you generated' in second_prompt

    @patch('tasks.ai.api.generate_text')
    def test_three_invalid_lengths_give_up(self, mock_generate):
        mock_generate.return_value = 'Hi'
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) is None
        assert mock_generate.call_count == 3

    @pytest.mark.parametrize('response', ['Error: boom', 'AI Naming Skipped', None])
    @patch('tasks.ai.api.generate_text')
    def test_a_provider_error_is_not_retried(self, mock_generate, response):
        mock_generate.return_value = response
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) is None
        assert mock_generate.call_count == 1

    @patch('tasks.ai.api.generate_text')
    def test_the_song_list_is_truncated_to_the_configured_size(
        self, mock_generate, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_IN_AI_PROMPT', 1)
        mock_generate.return_value = 'Velvet Morning Light'
        get_ai_playlist_title('Name it.', SONGS, ai_config())
        prompt = mock_generate.call_args[0][0]
        assert 'Vienna' in prompt and 'Skinny Love' not in prompt


def _call_helper(name='Rock_Fast_automatic', provider='OLLAMA'):
    return clustering_helper._try_ai_name_playlist(
        name,
        SONGS,
        {},
        provider,
        'http://localhost:11434/api/generate',
        'test-model',
        '', '', '', '', '', '', '',
    )


def _must_not_run(*_args, **_kwargs):
    raise AssertionError('this path must not run')


class TestClusteringNamingStyles:
    def test_title_style_returns_the_ai_title_without_touching_the_database(
        self, monkeypatch
    ):
        received = {}

        def fake_title(instructions, songs, config_dict):
            received.update(instructions=instructions, songs=songs, provider=config_dict['provider'])
            return 'Velvet Morning Light\n'

        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(config, 'AI_NAMING_TITLE_PROMPT', 'Edited instructions.')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', fake_title)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_lyrics_axis_vectors', _must_not_run)

        assert _call_helper() == 'Velvet Morning Light'
        assert received == {
            'instructions': 'Edited instructions.', 'songs': SONGS, 'provider': 'OLLAMA',
        }

    def test_title_style_failure_keeps_the_tag_based_name(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', lambda *_a: None)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', _must_not_run)
        assert _call_helper('Rock_Fast_automatic') == 'Rock_Fast_automatic'

    def test_default_style_never_calls_the_title_path(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'concept')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', lambda _ids: [])
        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', lambda *a, **k: 'Joyful Soul')
        assert _call_helper() == 'Joyful Soul'

    def test_an_unknown_style_value_behaves_like_the_default(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'garbage')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', lambda _ids: [])
        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', lambda *a, **k: 'Joyful Soul')
        assert _call_helper() == 'Joyful Soul'

    def test_no_provider_skips_both_styles(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', _must_not_run)
        assert _call_helper('Rock_Fast_automatic', provider='NONE') == 'Rock_Fast_automatic'
