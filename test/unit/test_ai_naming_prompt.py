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
* get_ai_playlist_title keeps the old 5-40 character contract with length feedback
  and no temperature override, retries a transient provider error, honours
  AI_NAMING_MAX_ATTEMPTS, and rejects a title another playlist already uses.
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
    ('1', 'Song Alpha', 'Artist A'),
    ('2', 'Song Beta', 'Artist B'),
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
            song_list_sample='- Song Alpha by Artist A\n- Song Beta by Artist B'
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
        assert block == TITLE_PROMPT_PLAYLIST_HEADER + '- Song Alpha by Artist A\n\n'

    def test_a_cap_below_one_still_sends_one_song(self):
        assert '- Song Alpha by Artist A' in title_prompt_song_block(SONGS, 0)

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
        assert prompt.count('- Song Alpha by Artist A') == 1

    def test_edited_instructions_cannot_remove_the_song_sample(self):
        prompt = build_title_naming_prompt('Ignore the songs.', SONGS, 5)
        assert prompt.endswith('- Song Alpha by Artist A\n- Song Beta by Artist B\n\n')


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
    def test_invalid_lengths_give_up_after_the_configured_attempts(self, mock_generate, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_MAX_ATTEMPTS', 4)
        mock_generate.return_value = 'Hi'
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) is None
        assert mock_generate.call_count == 4

    @pytest.mark.parametrize('response', ['Error: AI service is currently unavailable.', None])
    @patch('tasks.ai.api.generate_text')
    def test_a_transient_provider_error_is_retried(self, mock_generate, response):
        mock_generate.side_effect = [response, 'Velvet Morning Light']
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) == 'Velvet Morning Light'
        assert mock_generate.call_count == 2

    @patch('tasks.ai.api.generate_text')
    def test_persistent_provider_errors_give_up(self, mock_generate):
        mock_generate.return_value = 'Error: boom'
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) is None
        assert mock_generate.call_count == config.AI_NAMING_MAX_ATTEMPTS

    @patch('tasks.ai.api.generate_text')
    def test_a_skipped_provider_is_not_retried(self, mock_generate):
        mock_generate.return_value = 'AI Naming Skipped'
        assert get_ai_playlist_title('Name it.', SONGS, ai_config()) is None
        assert mock_generate.call_count == 1

    @patch('tasks.ai.api.generate_text')
    def test_an_already_used_title_is_rejected_and_retried(self, mock_generate):
        mock_generate.side_effect = ['late night drive', 'Velvet Morning Light']
        title = get_ai_playlist_title('Name it.', SONGS, ai_config(), used_titles=['Late Night Drive'])
        assert title == 'Velvet Morning Light'
        second_prompt = mock_generate.call_args_list[1][0][0]
        assert "The title 'late night drive' is already used" in second_prompt

    @patch('tasks.ai.api.generate_text')
    def test_only_used_titles_fall_back_to_the_used_spelling_instead_of_the_tag_name(
        self, mock_generate, monkeypatch
    ):
        monkeypatch.setattr(config, 'AI_NAMING_MAX_ATTEMPTS', 3)
        mock_generate.side_effect = ['late night drive', 'Hi', 'Error: boom']
        title = get_ai_playlist_title('Name it.', SONGS, ai_config(), used_titles=['Late Night Drive'])
        assert title == 'Late Night Drive', (
            'the exact used spelling lets the run add its (2) suffix, like a duplicate '
            'title did before duplicates were retried'
        )
        assert mock_generate.call_count == 3

    @patch('tasks.ai.api.generate_text')
    def test_used_titles_reach_the_prompt_after_the_songs(self, mock_generate):
        mock_generate.return_value = 'Velvet Morning Light'
        get_ai_playlist_title('Name it.', SONGS, ai_config(), used_titles=['Late Night Drive', 'Rainy Days'])
        prompt = mock_generate.call_args[0][0]
        assert prompt.endswith('Titles already used, do not reuse them: Late Night Drive | Rainy Days\n\n')
        assert prompt.index('This is the playlist:') < prompt.index('Titles already used')

    @patch('tasks.ai.api.generate_text')
    def test_no_used_titles_keep_the_version_2_render(self, mock_generate):
        mock_generate.return_value = 'Velvet Morning Light'
        get_ai_playlist_title(config._AI_NAMING_TITLE_PROMPT_DEFAULT, SONGS, ai_config(), used_titles=[])
        assert 'Titles already used' not in mock_generate.call_args[0][0]

    @patch('tasks.ai.api.generate_text')
    def test_the_song_list_is_truncated_to_the_configured_size(
        self, mock_generate, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_IN_AI_PROMPT', 1)
        mock_generate.return_value = 'Velvet Morning Light'
        get_ai_playlist_title('Name it.', SONGS, ai_config())
        prompt = mock_generate.call_args[0][0]
        assert 'Song Alpha' in prompt and 'Song Beta' not in prompt


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

        def fake_title(instructions, songs, config_dict, used_titles=None):
            received.update(instructions=instructions, songs=songs, provider=config_dict['provider'],
                            used_titles=used_titles)
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
            'used_titles': [],
        }

    def test_title_style_failure_keeps_the_tag_based_name(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', lambda *_a, **_k: None)
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

    def test_an_explicit_title_style_and_prompt_override_the_saved_ones(self, monkeypatch):
        received = {}

        def fake_title(instructions, songs, config_dict, used_titles=None):
            received['instructions'] = instructions
            return 'Velvet Morning Light'

        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'concept')
        monkeypatch.setattr(config, 'AI_NAMING_TITLE_PROMPT', 'Saved instructions.')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', fake_title)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', _must_not_run)
        name = clustering_helper._try_ai_name_playlist(
            'Rock_Fast_automatic', SONGS, {}, 'OLLAMA', 'u', 'm', '', '', '', '', '', '', '',
            naming_mode='title', title_prompt='Unsaved instructions.',
        )
        assert name == 'Velvet Morning Light'
        assert received == {'instructions': 'Unsaved instructions.'}

    def test_an_explicit_default_style_overrides_a_saved_title_style(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', lambda _ids: [])
        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', lambda *a, **k: 'Joyful Soul')
        name = clustering_helper._try_ai_name_playlist(
            'Rock_Fast_automatic', SONGS, {}, 'OLLAMA', 'u', 'm', '', '', '', '', '', '', '',
            naming_mode='concept',
        )
        assert name == 'Joyful Soul'

    def test_a_title_style_without_an_explicit_prompt_uses_the_saved_one(self, monkeypatch):
        received = {}

        def fake_title(instructions, songs, config_dict, used_titles=None):
            received['instructions'] = instructions
            return 'Velvet Morning Light'

        monkeypatch.setattr(config, 'AI_NAMING_TITLE_PROMPT', 'Saved instructions.')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', fake_title)
        clustering_helper._try_ai_name_playlist(
            'Rock_Fast_automatic', SONGS, {}, 'OLLAMA', 'u', 'm', '', '', '', '', '', '', '',
            naming_mode='title',
        )
        assert received == {'instructions': 'Saved instructions.'}

    def test_the_read_transaction_ends_before_the_ai_call(self, monkeypatch):
        from flask import g

        from flask_app import app

        order = []

        class FakeConn:
            closed = 0

            def commit(self):
                order.append('commit')

        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'concept')
        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', lambda ids: order.append('read') or [])
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name',
                            lambda *a, **k: order.append('ai') or 'Joyful Soul')
        with app.app_context():
            g.db = FakeConn()
            assert _call_helper() == 'Joyful Soul'
            del g.db
        assert order == ['read', 'commit', 'ai']

    def test_no_app_context_means_no_transaction_to_end(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'concept')
        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)
        monkeypatch.setattr(clustering_helper, 'get_score_data_by_ids', lambda ids: [])
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', lambda *a, **k: 'Joyful Soul')
        assert _call_helper() == 'Joyful Soul'

    def test_title_style_passes_the_used_titles_without_tag_names(self, monkeypatch):
        received = {}

        def fake_title(instructions, songs, config_dict, used_titles=None):
            received['used_titles'] = used_titles
            return 'Velvet Morning Light'

        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', fake_title)
        clustering_helper._try_ai_name_playlist(
            'Rock_Fast_automatic', SONGS, {}, 'OLLAMA', 'u', 'm', '', '', '', '', '', '', '',
            ['Late Night Drive', 'Rock_Pop_Fast_automatic', 'Rainy Days'],
        )
        assert received['used_titles'] == ['Late Night Drive', 'Rainy Days']

    def test_no_provider_skips_both_styles(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_NAMING_PROMPT_MODE', 'title')
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_title', _must_not_run)
        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', _must_not_run)
        assert _call_helper('Rock_Fast_automatic', provider='NONE') == 'Rock_Fast_automatic'


class TestOnlyShownTitlesAreRejected:
    @patch('tasks.ai.api.generate_text')
    def test_a_title_used_long_ago_and_not_shown_is_accepted_at_once(self, mock_generate):
        from tasks.ai.prompts import TITLE_PROMPT_RECENT_TITLES

        old = ['Old Title Number %d' % i for i in range(TITLE_PROMPT_RECENT_TITLES + 5)]
        mock_generate.return_value = 'Old Title Number 0'
        title = get_ai_playlist_title('Name it.', SONGS, ai_config(), used_titles=old)
        assert title == 'Old Title Number 0'
        assert mock_generate.call_count == 1, 'the model is only refused titles it was shown'
