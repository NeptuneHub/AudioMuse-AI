# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guards for the setup wizard's AI Prompt section.

There is no javascript runner in this repo, so the markup and setup.js are pinned by
reading them as sources, while the backend helpers are called directly. The section
offers exactly two naming styles, edits only the full-title instructions, shows the
appended song list as a read-only example, and starts a titles preview.

Main Features:
* The collapsed section sits after Lyrics API, inside the form, outside the lyrics gate
* Only the naming style and the title instructions are form fields; there is no
  variable palette and the example song block is never submitted
* MAX_SONGS_IN_AI_PROMPT stays in the advanced list where it always lived
* Save validation rejects an unknown style and an empty or oversized prompt, and
  normalises Windows line endings so Reset to default compares equal
* The preview route refuses to start without a provider or while one is running
"""

import os
import re

import app_setup

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

PROMPT_FIELDS = ('AI_NAMING_PROMPT_MODE', 'AI_NAMING_TITLE_PROMPT')


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _template():
    return _read(os.path.join('templates', 'setup.html'))


def _setup_js():
    return _read(os.path.join('static', 'setup.js'))


def _section():
    markup = _template()
    start = markup.index('<details id="ai-prompt-config">')
    return markup[start:markup.index('</details>', start)]


class TestSectionMarkup:
    def test_the_section_is_collapsed_by_default(self):
        match = re.search(r'<details id="ai-prompt-config"([^>]*)>', _template())
        assert match and 'open' not in match.group(1)

    def test_the_section_sits_after_lyrics_api_and_before_the_save_button(self):
        markup = _template()
        assert (
            markup.index('<details id="lyrics-api-config">')
            < markup.index('<details id="ai-prompt-config">')
            < markup.index('id="save-button"')
        )

    def test_the_section_is_inside_the_form(self):
        markup = _template()
        assert (
            markup.index('<form id="setup-form"')
            < markup.index('<details id="ai-prompt-config">')
            < markup.index('</form>')
        )

    def test_the_section_is_not_gated_on_lyrics_being_enabled(self):
        markup = _template()
        lyrics_end = markup.index('</details>', markup.index('<details id="lyrics-api-config">'))
        between = markup[lyrics_end:markup.index('<details id="ai-prompt-config">')]
        assert '{% endif %}' in between and '{% if ' not in between

    def test_exactly_two_naming_styles_are_offered(self):
        section = _section()
        select = section[section.index('id="AI_NAMING_PROMPT_MODE"'):]
        select = select[:select.index('</select>')]
        assert re.findall(r'value="(\w+)"', select) == ['concept', 'title']

    def test_the_options_carry_only_a_short_name(self):
        section = _section()
        select = section[section.index('id="AI_NAMING_PROMPT_MODE"'):]
        select = select[:select.index('</select>')]
        assert re.findall(r'<option value="\w+">([^<]*)</option>', select) == [
            'Word + genre', 'Full AI title',
        ]

    def test_each_style_has_its_own_explanation_and_only_the_default_starts_visible(self):
        section = _section()
        concept = re.search(r'<p id="ai-prompt-concept-help"([^>]*)>(.*?)</p>', section, re.S)
        title = re.search(r'<p id="ai-prompt-title-help"([^>]*)>(.*?)</p>', section, re.S)
        assert concept and title
        assert 'display:none' not in concept.group(1)
        assert 'display:none' in title.group(1)
        assert 'Joyful Soul' in concept.group(2)
        assert 'Raindrops on the Windshield' in title.group(2)

    def test_the_preview_is_available_for_both_styles(self):
        section = _section()
        panel_start = section.index('<div id="ai-prompt-title-panel"')
        depth = 0
        panel_end = None
        for match in re.finditer(r'<div\b|</div>', section[panel_start:]):
            depth += 1 if match.group(0) == '<div' else -1
            if depth == 0:
                panel_end = panel_start + match.end()
                break
        panel = section[panel_start:panel_end]
        assert 'ai-prompt-song-block' in panel
        for element_id in ('ai-prompt-preview-start', 'ai-prompt-preview-status',
                           'ai-prompt-preview-titles'):
            assert 'id="%s"' % element_id not in panel, element_id
            assert section.index('id="%s"' % element_id) > panel_end

    def test_only_the_style_and_title_instructions_are_form_fields(self):
        section = _section()
        names = re.findall(r'\bname="([A-Z_]+)"', section)
        assert sorted(names) == sorted(PROMPT_FIELDS)
        for field in PROMPT_FIELDS:
            assert 'id="%s" name="%s"' % (field, field) in section

    def test_there_is_no_variable_palette_nor_concept_prompt(self):
        section = _section()
        assert 'ai-prompt-tokens' not in section
        assert 'AI_NAMING_CONCEPT_PROMPT' not in _template()
        assert '<textarea' in section and section.count('<textarea') == 1

    def test_the_song_block_is_a_read_only_example(self):
        section = _section()
        block = re.search(r'<pre id="ai-prompt-song-block"[^>]*>', section)
        assert block and 'name=' not in block.group(0)

    def test_the_preview_controls_exist(self):
        section = _section()
        for element_id in ('ai-prompt-reset', 'ai-prompt-preview-start', 'ai-prompt-preview-stop',
                           'ai-prompt-preview-status', 'ai-prompt-preview-titles'):
            assert 'id="%s"' % element_id in section, element_id


class TestAdvancedListUnchanged:
    def test_max_songs_stays_in_the_ai_advanced_group(self):
        source = _setup_js()
        sections = source[source.index('ADVANCED_SECTIONS'):source.index('ADVANCED_OTHER_TITLE')]
        assert "'MAX_SONGS_IN_AI_PROMPT'" in sections
        assert 'MAX_SONGS_IN_AI_PROMPT' not in _template()
        assert 'MAX_SONGS_IN_AI_PROMPT' not in app_setup.HIDDEN_ADVANCED_FIELDS

    def test_the_prompt_fields_are_hidden_from_advanced_and_allowed_for_saving(self):
        for field in PROMPT_FIELDS:
            assert field in app_setup.HIDDEN_ADVANCED_FIELDS
            assert not app_setup.should_show_advanced(field)
        assert set(app_setup.AI_PROMPT_CONFIG_FIELDS) == set(PROMPT_FIELDS)
        assert 'allowed_keys.update(AI_PROMPT_CONFIG_FIELDS)' in _read('app_setup.py')


class TestScript:
    def test_fields_are_not_submitted_before_the_section_is_loaded(self):
        source = _setup_js()
        collect = source[source.index('function collectConfigFromForm'):]
        collect = collect[:collect.index('\n}')]
        assert 'aiPromptState.loaded' in collect

    def test_the_payload_is_populated_on_load(self):
        assert 'populateAiPromptFields(data.ai_prompt_fields)' in _setup_js()

    def test_reset_clears_the_original_value_marker(self):
        source = _setup_js()
        reset = source[source.index('function resetAiPromptToDefault'):]
        reset = reset[:reset.index('\n}')]
        assert 'delete area.dataset.originalValue' in reset

    def test_the_preview_posts_the_selected_style_and_unsaved_instructions_and_polls(self):
        source = _setup_js()
        start = source[source.index('function startAiPromptPreview'):]
        start = start[:start.index('\nfunction ')]
        assert "'/api/setup/ai-prompt/preview'" in start
        assert 'mode: aiPromptMode()' in start
        assert 'area.value' in start
        assert 'pollAiPromptPreview()' in start

    def test_switching_style_toggles_both_explanations(self):
        source = _setup_js()
        toggle = source[source.index('function updateAiPromptMode'):]
        toggle = toggle[:toggle.index('\n}')]
        for element_id in ('ai-prompt-concept-help', 'ai-prompt-title-help', 'ai-prompt-title-panel'):
            assert element_id in toggle, element_id

    def test_the_start_button_is_disabled_before_the_request(self):
        source = _setup_js()
        start = source[source.index('function startAiPromptPreview'):]
        start = start[:start.index('\nfunction ')]
        assert start.index('button.disabled = true') < start.index("fetch('/api/setup/ai-prompt/preview'")

    def test_a_conflict_from_a_running_preview_starts_polling(self):
        source = _setup_js()
        start = source[source.index('function startAiPromptPreview'):]
        start = start[:start.index('\nfunction ')]
        assert 'result.body.preview_running' in start and 'pollAiPromptPreview()' in start

    def test_polling_checks_the_response_status(self):
        source = _setup_js()
        poll = source[source.index('function pollAiPromptPreview'):]
        poll = poll[:poll.index('\nfunction ')]
        assert 'readAiPromptResponse' in poll and '!result.ok' in poll
        assert 'AI_PROMPT_MAX_POLL_FAILURES' in poll

    def test_giving_up_on_polling_re_enables_start_on_both_failure_paths(self):
        source = _setup_js()
        poll = source[source.index('function pollAiPromptPreview'):]
        poll = poll[:poll.index('\nfunction ')]
        assert poll.count('releaseAiPromptPreview(AI_PROMPT_POLL_GAVE_UP_MESSAGE)') == 2
        release = source[source.index('function releaseAiPromptPreview'):]
        release = release[:release.index('\nfunction ')]
        assert 'button.disabled = false' in release
        assert "stop.style.display = 'none'" in release

    def test_stop_runs_the_global_cancel_for_the_preview(self):
        source = _setup_js()
        stop = source[source.index('function stopAiPromptPreview'):]
        stop = stop[:stop.index('\nfunction ')]
        assert "fetch('/api/cancel/' + encodeURIComponent(aiPromptState.taskId)" in stop
        assert 'id="ai-prompt-preview-stop"' in _section()

    def test_titles_are_rendered_as_text_not_html(self):
        source = _setup_js()
        render = source[source.index('function renderAiPromptPreview'):]
        render = render[:render.index('\nfunction ')]
        assert 'name.textContent = entry.title' in render
        assert 'innerHTML = entry' not in render


class TestBackendValidation:
    def test_an_unknown_style_is_rejected(self):
        assert app_setup._validate_ai_prompt_values({'AI_NAMING_PROMPT_MODE': 'poem'})

    def test_the_style_is_normalised(self):
        values = {'AI_NAMING_PROMPT_MODE': ' Title '}
        assert app_setup._validate_ai_prompt_values(values) is None
        assert values['AI_NAMING_PROMPT_MODE'] == 'title'

    def test_an_empty_prompt_is_rejected(self):
        assert app_setup._validate_ai_prompt_values({'AI_NAMING_TITLE_PROMPT': '  \n '})

    def test_an_oversized_prompt_is_rejected(self):
        text = 'x' * (app_setup.AI_TITLE_PROMPT_MAX_CHARS + 1)
        assert app_setup._validate_ai_prompt_values({'AI_NAMING_TITLE_PROMPT': text})

    def test_windows_line_endings_are_normalised(self):
        values = {'AI_NAMING_TITLE_PROMPT': 'Line one\r\nLine two\r\n'}
        assert app_setup._validate_ai_prompt_values(values) is None
        assert values['AI_NAMING_TITLE_PROMPT'] == 'Line one\nLine two\n'

    def test_untouched_fields_pass(self):
        assert app_setup._validate_ai_prompt_values({'OTHER': 'x'}) is None

    def test_the_payload_carries_the_default_and_an_example_song_block(self):
        payload = app_setup._build_ai_prompt_payload()
        assert payload['mode'] in ('concept', 'title')
        assert payload['title_prompt_default'] == app_setup.config._AI_NAMING_TITLE_PROMPT_DEFAULT
        assert payload['example_song_block'].startswith('This is the playlist:\n- ')
        assert payload['preview_max_songs'] == 10000


class TestPreviewRoute:
    def _post(self, body):
        with app_setup.app.test_request_context(
            '/api/setup/ai-prompt/preview', method='POST', json=body
        ):
            response = app_setup.setup_ai_prompt_preview()
        return response

    def test_no_provider_refuses_to_start(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'NONE')
        started = []
        monkeypatch.setattr(app_setup.naming_preview, 'start_preview', lambda *a: started.append(a))
        body, status = self._post({'instructions': 'Name it.'})
        assert status == 400 and not started

    def test_an_empty_prompt_refuses_to_start_the_title_style(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')
        body, status = self._post({'mode': 'title', 'instructions': '   '})
        assert status == 400

    def test_a_running_preview_returns_conflict_and_says_so(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')
        monkeypatch.setattr(app_setup.naming_preview, 'start_preview',
                            lambda *a: (None, app_setup.naming_preview.PREVIEW_RUNNING_MESSAGE))
        body, status = self._post({'mode': 'title', 'instructions': 'Name it.'})
        assert status == 409 and body.get_json()['preview_running'] is True

    def test_another_batch_task_returns_conflict_with_the_parallel_rule(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')
        busy = app_setup.naming_preview.PREVIEW_BUSY_MESSAGE.format(task_type='main_clustering')
        monkeypatch.setattr(app_setup.naming_preview, 'start_preview', lambda *a: (None, busy))
        body, status = self._post({'mode': 'concept'})
        payload = body.get_json()
        assert status == 409 and payload['preview_running'] is False
        assert 'never runs two batch tasks in parallel' in payload['error']

    def _capture_start(self, monkeypatch):
        received = {}

        def fake_start(mode, instructions):
            received.update(mode=mode, instructions=instructions)
            return 'job-1', None

        monkeypatch.setattr(app_setup.naming_preview, 'start_preview', fake_start)
        return received

    def test_a_title_start_queues_the_normalised_text_and_never_a_client_provider(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'ollama')
        received = self._capture_start(monkeypatch)
        body, status = self._post({'mode': 'title', 'instructions': 'A\r\nB', 'provider': 'GEMINI'})
        assert status == 202
        assert received == {'mode': 'title', 'instructions': 'A\nB'}
        assert body.get_json()['task_id'] == 'job-1'
        assert body.get_json()['status'] == 'running'

    def test_the_default_style_starts_without_a_prompt(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')
        received = self._capture_start(monkeypatch)
        body, status = self._post({'mode': 'concept', 'instructions': ''})
        assert status == 202
        assert received == {'mode': 'concept', 'instructions': None}

    def test_a_queue_error_is_a_generic_500(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')

        def boom(mode, instructions):
            raise RuntimeError('secret database detail')

        monkeypatch.setattr(app_setup.naming_preview, 'start_preview', boom)
        body, status = self._post({'mode': 'concept'})
        assert status == 500
        assert 'secret' not in body.get_json()['error']

    def test_the_status_read_is_a_generic_500_on_error(self, monkeypatch):
        def boom():
            raise RuntimeError('secret database detail')

        monkeypatch.setattr(app_setup.naming_preview, 'preview_status', boom)
        with app_setup.app.test_request_context('/api/setup/ai-prompt/preview', method='GET'):
            body, status = app_setup.setup_ai_prompt_preview()
        assert status == 500
        assert 'secret' not in body.get_json()['error']

    def test_a_missing_or_unknown_style_previews_the_default(self, monkeypatch):
        monkeypatch.setattr(app_setup.config, 'AI_MODEL_PROVIDER', 'OLLAMA')
        received = self._capture_start(monkeypatch)
        body, status = self._post({'mode': 'poem'})
        assert status == 202
        assert received['mode'] == 'concept'
