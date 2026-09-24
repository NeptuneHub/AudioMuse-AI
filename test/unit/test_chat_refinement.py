# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chat request parsing, library-calibrated energy and the in-range re-rank tiers.

Covers the deterministic layer that lets a small planner model build faithful
playlists: request-wording hints, plan repairs, the calibrated energy scale,
and the re-rank dimensions added for explicit ranges.

Main Features:
* Hints: BPM bounds, multilingual negation, decade modifiers and foreign decades, relative eras, key/scale, negated instrumentals, track length and playlist time budget, per-artist caps, song counts, excluded versions, sound words and hyphenated compounds
* Plan repairs: point ranges widened, excluded and broader genres dropped, relative eras and instrumental wording override the model, sound words add an audio finder, unrated libraries drop min_rating, a journey cue fixes blend_mode and runs alone, seed-relative energy/tempo, genre backfill
* Energy 0..1 maps to library percentiles, with the configured linear scale as fallback
* Re-rank: explicit year, BPM, track length and recently-added ranges rank in-range songs first, the genre leads the other tiers, a mid range never scores flat, very short tracks sink; filter-only genre results lead with main-style songs
* Album lookup ignores the artist's own name, and search_database adds the duration and recently-added SQL bounds
* Instruments: words map to SAE concepts, a metadata-only plan gets a steered sound search, genre plus instrument rank first, concept percentiles come from the library sample, a copied prompt example is replaced
"""

import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

import config
from tasks.ai import calibration, planner, rerank, tool_impl
from tasks.ai.planner import ToolPlan, extract_hints


def _hints(text, **kwargs):
    h = extract_hints(text, **kwargs)
    h.pop('notes', None)
    return h


class TestTempoAndKeyHints:
    @pytest.mark.parametrize(
        'text, expected',
        [
            ('tracks over 140 bpm', {'tempo_min': 140.0}),
            ('music under 90 bpm', {'tempo_max': 90.0}),
            ('120-130 bpm grooves', {'tempo_min': 120.0, 'tempo_max': 130.0}),
        ],
    )
    def test_bpm_bounds_are_explicit(self, text, expected):
        h = _hints(text)
        for key, value in expected.items():
            assert h[key] == value
        assert h['tempo_explicit'] is True

    def test_a_point_bpm_stays_a_point_hint(self):
        h = _hints('songs around 170 bpm')
        assert h['bpm'] == 170 and h['tempo_explicit'] is True

    def test_activity_words_set_a_tempo_floor(self):
        h = _hints('workout songs')
        assert h['tempo_min'] >= 115 and h['energy_min'] >= 0.6

    @pytest.mark.parametrize(
        'text, key, scale',
        [
            ('songs in C major', 'C', 'major'),
            ('Bb minor jazz', 'Bb', 'minor'),
            ('songs in the key of F#', 'F#', None),
        ],
    )
    def test_key_and_scale(self, text, key, scale):
        h = _hints(text)
        assert h.get('key') == key
        assert h.get('scale') == scale

    def test_an_article_a_is_not_a_key(self):
        h = _hints('a minor thing for me')
        assert 'key' not in h


class TestNegationAndGenreHints:
    @pytest.mark.parametrize(
        'text, excluded',
        [
            ('niente rap, solo rock', 'Hip-Hop'),
            ('musica senza jazz', 'jazz'),
            ('pas de metal', 'metal'),
            ('musik ohne rap', 'Hip-Hop'),
        ],
    )
    def test_negation_in_other_languages(self, text, excluded):
        assert excluded in _hints(text).get('exclude_genres', [])

    def test_non_stop_is_not_a_negation(self):
        h = _hints('non-stop dance music')
        assert 'dance' in h.get('genres', []) and not h.get('exclude_genres')

    def test_a_hyphenated_compound_does_not_leak_its_parent_genre(self):
        h = _hints('instrumental post-rock')
        assert 'rock' not in h.get('genres', [])
        assert 'post-rock' in h['sound_words']

    def test_a_version_exclusion_does_not_negate_the_genre_after_it(self):
        h = _hints('no live versions of classic rock')
        assert h['exclude_versions'] == ['live']
        assert 'classic rock' in h['genres'] and not h.get('exclude_genres')

    def test_adjective_genres(self):
        assert set(_hints('soulful funky grooves')['genres']) == {'soul', 'funk'}

    def test_negated_sound_words_are_skipped(self):
        assert 'reggaeton' not in _hints('rock sin reggaeton').get('sound_words', [])

    def test_accents_are_folded_for_sound_words(self):
        assert 'reggaeton' in _hints('musica para bailar reggaet\u00f3n')['sound_words']


class TestEraHints:
    @pytest.mark.parametrize(
        'text, lo, hi',
        [
            ('early 90s hip hop', 1990, 1994),
            ('late 80s pop', 1985, 1989),
            ("90's rock", 1990, 1999),
            ('the nineties', 1990, 1999),
            ('rock anni 90', 1990, 1999),
            ('80er pop', 1980, 1989),
        ],
    )
    def test_decades(self, text, lo, hi):
        h = _hints(text)
        assert (h['year_min'], h['year_max']) == (lo, hi)

    def test_last_n_years_is_relative_to_today(self):
        year = datetime.date.today().year
        h = _hints('hits from the last 5 years')
        assert (h['year_min'], h['year_max']) == (year - 4, year)
        assert h['year_relative'] is True

    def test_recent_uses_the_newest_library_year(self):
        h = _hints('recent songs', library_year_max=2020)
        assert (h['year_min'], h['year_max']) == (2018, 2020)


class TestShapeHints:
    def test_instrumental_negation(self):
        assert _hints('no instrumentals please')['instrumental'] is False
        assert _hints('songs without vocals')['instrumental'] is True

    @pytest.mark.parametrize(
        'text, key, seconds',
        [
            ('songs under 3 minutes', 'duration_max', 180.0),
            ('tracks longer than 6 minutes', 'duration_min', 360.0),
            ('short songs', 'duration_max', planner.SHORT_TRACK_SECONDS),
            ('long epic songs', 'duration_min', planner.LONG_TRACK_SECONDS),
        ],
    )
    def test_track_length(self, text, key, seconds):
        assert _hints(text)[key] == seconds

    def test_a_per_track_bound_is_not_a_time_budget(self):
        h = _hints('songs under 5 minutes long')
        assert h['duration_max'] == 300.0 and 'total_seconds' not in h

    @pytest.mark.parametrize(
        'text, seconds',
        [('an hour of chill music', 3600.0), ('a 45-minute workout mix', 2700.0), ('2 hours of jazz', 7200.0)],
    )
    def test_time_budget(self, text, seconds):
        assert _hints(text)['total_seconds'] == seconds

    @pytest.mark.parametrize(
        'text, cap',
        [('one song per artist', 1), ('max 2 tracks per artist', 2), ('one hit wonders', 1)],
    )
    def test_per_artist_cap(self, text, cap):
        assert _hints(text)['max_per_artist'] == cap

    def test_song_count_ignores_per_artist_phrases(self):
        assert _hints('give me 15 songs of rock')['song_count'] == 15
        assert 'song_count' not in _hints('1 song per artist')

    def test_studio_only_excludes_live_and_demo(self):
        assert _hints('studio versions only')['exclude_versions'] == ['live', 'demo']

    def test_recently_added(self):
        assert _hints('songs added recently')['added_within_days'] == planner.RECENTLY_ADDED_DAYS
        assert _hints('songs added this week')['added_within_days'] == 7

    def test_requested_playlist_shape(self):
        shape = planner.requested_playlist_shape('give me 20 songs, one per artist')
        assert shape == {'song_count': 20, 'max_per_artist': 1}


class TestPlanRepairs:
    def test_a_point_tempo_is_widened(self):
        notes = []
        filt = planner._normalize_filter_inplace({'tempo_min': 170, 'tempo_max': 170}, notes)
        assert (filt['tempo_min'], filt['tempo_max']) == (160.0, 180.0)
        assert notes

    def test_an_intentional_narrow_range_is_kept(self):
        filt = planner._normalize_filter_inplace({'tempo_min': 120, 'tempo_max': 130}, [])
        assert (filt['tempo_min'], filt['tempo_max']) == (120.0, 130.0)

    def test_unrequested_duration_and_recency_are_stripped(self):
        plan = ToolPlan(filter={'genres': ['rock'], 'duration_max': 300, 'added_within_days': 30})
        planner._strip_unrequested_filter_args(plan, {}, '90s rock with a female singer', [])
        assert plan.filter == {'genres': ['rock']}

    def test_requested_duration_survives(self):
        plan = ToolPlan(filter={'duration_max': 180})
        planner._strip_unrequested_filter_args(plan, {'duration_max': 180}, 'songs under 3 minutes', [])
        assert plan.filter == {'duration_max': 180}

    def test_between_bpm_is_a_range(self):
        h = _hints('tracks between 120 and 130 bpm')
        assert (h['tempo_min'], h['tempo_max']) == (120.0, 130.0)

    def test_slow_is_an_explicit_tempo_but_upbeat_is_not(self):
        assert _hints('very slow songs')['tempo_explicit'] is True
        assert 'tempo_explicit' not in _hints('upbeat songs')

    def test_a_specific_genre_replaces_its_parent(self):
        notes = []
        filt = planner._normalize_filter_inplace({'genres': ['rock', 'Progressive rock', 'jazz']}, notes)
        assert filt['genres'] == ['Progressive rock', 'jazz'] and notes
        plan = ToolPlan(filter={'genres': ['rock']})
        planner._apply_hint_backstop(plan, {'genres': ['Progressive rock']}, [])
        assert plan.filter['genres'] == ['Progressive rock']

    def test_excluded_genres_the_request_never_names_are_dropped(self):
        plan = ToolPlan(filter={'instrumental': True, 'exclude_genres': ['pop', 'rock', 'Hip-Hop']})
        planner._strip_contradictory_exclusions(plan, {}, 'calm songs to sleep, sans paroles, no rappers', [])
        assert plan.filter == {'instrumental': True, 'exclude_genres': ['Hip-Hop']}

    def test_the_backstop_never_adds_an_excluded_genre(self):
        plan = ToolPlan(filter={'exclude_genres': ['Hip-Hop']})
        planner._apply_hint_backstop(plan, {'genres': ['Hip-Hop', 'rock']}, [])
        assert plan.filter['genres'] == ['rock']

    def test_relative_era_and_instrumental_wording_win(self):
        plan = ToolPlan(filter={'year_min': 2019, 'year_max': 2020, 'instrumental': True})
        hints = {'year_min': 2024, 'year_max': 2026, 'year_relative': True, 'instrumental': False}
        planner._apply_hint_backstop(plan, hints, [])
        assert (plan.filter['year_min'], plan.filter['year_max']) == (2024, 2026)
        assert plan.filter['instrumental'] is False

    def test_an_explicit_bpm_marks_the_tempo_exact(self):
        plan = ToolPlan(filter={'genres': ['rock']})
        planner._apply_hint_backstop(plan, {'bpm': 170, 'tempo_explicit': True}, [])
        assert plan.filter[rerank.EXACT_TEMPO_KEY] is True
        assert (plan.filter['tempo_min'], plan.filter['tempo_max']) == (160.0, 180.0)

    def test_sound_words_add_an_audio_finder(self, monkeypatch):
        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        plan = ToolPlan(filter={'genres': ['electronic']})
        planner._add_sound_primary(plan, {'sound_words': ['dark', 'moody']}, {'text_match'}, [])
        assert plan.primaries == [
            {'name': 'text_match', 'arguments': {'query': 'dark moody electronic music', 'mode': 'audio'}}
        ]

    def test_a_mood_only_filter_stays_a_metadata_search(self, monkeypatch):
        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        plan = ToolPlan(filter={'moods': ['sad'], 'energy_max': 0.3})
        planner._add_sound_primary(plan, {}, {'text_match'}, [])
        assert plan.primaries == []

    def test_a_pool_short_of_genre_matches_is_backfilled_from_the_library(self, monkeypatch):
        import tasks.ai.tools as tools_mod

        pool = [{'item_id': 'p1'}, {'item_id': 'p2'}]
        feats = {'p1': {'mood_vector': 'rock:0.6'}, 'p2': {'mood_vector': 'pop:0.6'}}
        calls = []

        def fake_exec(name, args, cfg):
            calls.append(args)
            return {'songs': [{'item_id': 'p1'}, {'item_id': 'x1'}]}

        monkeypatch.setattr(tools_mod, 'execute_mcp_tool', fake_exec)
        monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {i: {'mood_vector': 'rock:0.7'} for i in ids})
        out = planner._backfill_genre_matches({'genres': ['rock']}, pool, feats, {}, 5, [])
        assert [s['item_id'] for s in out] == ['p1', 'p2', 'x1']
        assert calls[0]['genres'] == ['rock'] and calls[0]['get_songs'] == 200

    def test_songs_whose_main_style_is_the_genre_lead(self, monkeypatch):
        songs = [{'item_id': 'x'}, {'item_id': 'y'}, {'item_id': 'z'}]
        tags = {
            'x': 'Hip-Hop:0.57,indie:0.53',
            'y': 'indie:0.58,rock:0.52',
            'z': 'electronic:0.6,pop:0.55,indie:0.51',
        }
        monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {i: {'mood_vector': tags[i]} for i in ids})
        out = planner._genre_purity_sort(songs, ['indie'], [])
        assert [s['item_id'] for s in out] == ['y', 'x', 'z']

    def test_filter_only_results_sink_very_short_tracks(self, monkeypatch):
        songs = [{'item_id': 's', 'title': 'Song 1'}, {'item_id': 'l', 'title': 'Song 2'}]
        monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {'s': {'duration': 40.0}, 'l': {'duration': 200.0}})
        planner._demote_non_songs(songs, [], 60.0)
        assert [s['item_id'] for s in songs] == ['l', 's']

    def test_no_backfill_without_a_genre(self, monkeypatch):
        pool = [{'item_id': 'p1'}]
        assert planner._backfill_genre_matches({'year_min': 1990}, pool, {}, {}, 5, []) == pool

    def test_an_artist_filter_keeps_its_own_songs(self, monkeypatch):
        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        plan = ToolPlan(filter={'artist': 'Artist A'})
        planner._add_sound_primary(plan, {'sound_words': ['dark']}, {'text_match'}, [])
        assert plan.primaries == []

    def test_an_unrated_library_drops_min_rating(self):
        plan = ToolPlan(filter={'min_rating': 4, 'genres': ['rock']})
        planner._strip_unrated_filter(plan, {'has_ratings': False}, [])
        assert 'min_rating' not in plan.filter and plan.notes

    def test_excluded_versions_are_removed_unless_nothing_is_left(self):
        songs = [
            {'item_id': '1', 'title': 'Song 1 (Live)', 'album': ''},
            {'item_id': '2', 'title': 'Song 2', 'album': 'Live at the Arena'},
            {'item_id': '3', 'title': 'Song 3', 'album': 'Album X'},
        ]
        assert [s['item_id'] for s in planner._drop_versions(songs, ['live'], [])] == ['3']
        assert planner._drop_versions(songs[:1], ['live'], []) == songs[:1]


class TestSeedRelative:
    def _plan(self):
        return ToolPlan(
            primaries=[{'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'Artist A'}]}}],
            filter={'energy_min': 0.0, 'energy_max': 0.35},
        )

    def test_more_chill_is_below_the_seed_energy(self, monkeypatch):
        monkeypatch.setattr(tool_impl, '_seed_profile', lambda seeds: {'energy': 0.7, 'tempo': 120.0})
        monkeypatch.setattr(calibration, 'energy_to_norm', lambda raw: 0.6)
        plan = self._plan()
        planner._apply_seed_relative(plan, 'like Artist A but more chill and slower', {}, [])
        assert plan.filter == {'energy_max': 0.45, 'tempo_max': 110}

    def test_more_upbeat_is_above_the_seed_energy(self, monkeypatch):
        monkeypatch.setattr(tool_impl, '_seed_profile', lambda seeds: {'energy': 0.7, 'tempo': 120.0})
        monkeypatch.setattr(calibration, 'energy_to_norm', lambda raw: 0.6)
        plan = self._plan()
        planner._apply_seed_relative(plan, 'like Artist A but more upbeat', {}, [])
        assert plan.filter == {'energy_min': 0.75}

    def test_without_seeds_nothing_changes(self):
        plan = ToolPlan(filter={'energy_max': 0.35})
        planner._apply_seed_relative(plan, 'more chill songs', {}, [])
        assert plan.filter == {'energy_max': 0.35}

    def test_mood_words_in_several_languages(self):
        assert _hints('chill songs I added this month')['moods'] == ['relaxed']
        assert _hints('canzoni tristi anni 90')['moods'] == ['sad']
        assert 'moods' not in _hints('songs that are not sad')

    def test_a_filter_only_plan_gets_the_missing_mood(self):
        plan = ToolPlan(filter={'energy_max': 0.35})
        planner._apply_hint_backstop(plan, {'moods': ['relaxed']}, [])
        assert plan.filter['moods'] == ['relaxed']

    def test_dance_needs_a_music_noun(self):
        assert 'dance' not in _hints('slow songs for our first dance').get('genres', [])
        assert _hints('dance music from the 90s')['genres'] == ['dance']
        assert _hints('Harte Rockmusik')['genres'] == ['rock']
        assert _hints('musica strumentale senza voce')['instrumental'] is True


class TestInstruments:
    @pytest.mark.parametrize(
        'text, instruments',
        [
            ('POP viola with female voice', ['viola']),
            ('groovy sax blues', ['saxophone']),
            ('tabla afrobeat fast-paced', ['percussion']),
            ('soft acoustic guitar for studying', ['acoustic guitar']),
        ],
    )
    def test_instrument_words_map_to_the_concept_catalogue(self, text, instruments):
        assert _hints(text)['instruments'] == instruments

    @pytest.mark.parametrize(
        'text, word',
        [('warm rhodes keys, mellow downtempo groove', 'rhodes'), ('dark heavy synth bass', 'synth bass'),
         ('gospel choir steelpan', 'choir')],
    )
    def test_timbres_the_concept_model_confuses_are_matched_by_sound_only(self, text, word):
        h = _hints(text)
        assert word in h['instrument_words'] and 'instruments' not in h

    def test_holiday_songs_go_last_unless_the_request_is_about_the_holidays(self):
        songs = [
            {'item_id': '1', 'title': 'Song 1', 'album': 'The Magic of Christmas'},
            {'item_id': '2', 'title': 'Song 2', 'album': 'Album X'},
        ]
        plan = ToolPlan()
        out = planner._shape_result({'songs': list(songs)}, _hints('pop viola with female voice'), [], plan)
        assert [s['item_id'] for s in out['songs']] == ['2', '1']
        out = planner._shape_result({'songs': list(songs)}, _hints('christmas pop with viola'), [], plan)
        assert [s['item_id'] for s in out['songs']] == ['1', '2']

    def test_a_negated_instrument_is_not_requested(self):
        assert 'instruments' not in _hints('chill songs with no drums')

    def _catalogue(self, monkeypatch, terms=('viola', 'piano')):
        import tasks.clap_steering as steering

        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        monkeypatch.setattr(steering, 'concept_terms', lambda: list(terms))

    def test_a_metadata_only_plan_gets_a_sound_search_and_the_instrument_check(self, monkeypatch):
        self._catalogue(monkeypatch)
        hints = _hints('POP viola with female voice')
        plan = ToolPlan(filter={'genres': ['pop'], 'voices': ['female vocalists']})
        planner._add_sound_primary(plan, hints, {'text_match'}, [])
        planner._apply_instruments(plan, hints, {'text_match'}, [])
        assert plan.primaries[0]['arguments']['query'] == 'viola pop music with female vocals'
        assert plan.filter['instruments'] == ['viola']
        assert planner._steering_for(plan.filter, 3.0) == [{'term': 'viola', 'weight': 3.0, 'direction': 'more'}]

    def test_an_instrument_outside_the_catalogue_is_left_to_the_sound_search(self, monkeypatch):
        self._catalogue(monkeypatch, terms=('piano',))
        plan = ToolPlan(primaries=[{'name': 'text_match', 'arguments': {'query': 'x'}}])
        planner._apply_instruments(plan, _hints('pop viola'), {'text_match'}, [])
        assert plan.filter is None

    def test_a_copied_prompt_example_is_replaced_by_the_request(self):
        calls = [{'name': 'text_match', 'arguments': {
            'query': 'dreamy rhodes keys, mellow downtempo groove', 'mode': 'audio'}}]
        out = planner.validate_plan_args(calls, user_wants_rating=False, request_text='dreamy rhodes metal')
        assert out[0]['arguments']['query'] == 'dreamy rhodes metal'

    def test_a_negation_only_sound_query_is_dropped(self):
        calls = [
            {'name': 'text_match', 'arguments': {'query': 'happy kids party', 'mode': 'audio'}},
            {'name': 'text_match', 'arguments': {'query': 'nothing explicit', 'mode': 'audio'}},
        ]
        out = planner.validate_plan_args(calls, user_wants_rating=False, request_text='happy songs, nothing explicit')
        assert [c['arguments']['query'] for c in out] == ['happy kids party']

    def test_a_translated_query_is_kept(self):
        calls = [{'name': 'text_match', 'arguments': {'query': 'energetic gym music', 'mode': 'audio'}}]
        out = planner.validate_plan_args(calls, user_wants_rating=False, request_text='musica energica per la palestra')
        assert out[0]['arguments']['query'] == 'energetic gym music'

    def test_genre_and_instrument_matches_rank_first(self):
        feats = {
            'pop_only': {'mood_vector': 'pop:0.6', 'instrument_pct': {'viola': 0.40}},
            'pop_viola': {'mood_vector': 'folk:0.6,pop:0.55', 'instrument_pct': {'viola': 0.98}},
            'viola_only': {'mood_vector': 'jazz:0.6', 'instrument_pct': {'viola': 0.99}},
        }
        pool = [{'item_id': k, 'title': 'Song 1'} for k in feats]
        final, _m, _mv = rerank.rerank(pool, {'genres': ['pop'], 'instruments': ['viola']}, feats, [])
        assert [s['item_id'] for s in final] == ['pop_viola', 'pop_only', 'viola_only']

    def test_the_instrument_check_runs_before_the_genre_backfill(self, monkeypatch):
        import tasks.ai.tools as tools_mod

        calls = []
        songs = [{'item_id': f's{i}', 'title': f'Song {i}', 'artist': f'Artist {i}'} for i in range(60)]

        def fake_exec(name, args, cfg):
            calls.append(name)
            return {'songs': songs if name == 'text_match' else [], 'message': ''}

        monkeypatch.setattr(tools_mod, 'execute_mcp_tool', fake_exec)
        monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {i: {'mood_vector': 'pop:0.6'} for i in ids})
        monkeypatch.setattr(calibration, 'concept_percentiles', lambda ids, terms: {i: {'viola': 0.99} for i in ids})
        plan = ToolPlan(
            primaries=[{'name': 'text_match', 'arguments': {'query': 'pop viola', 'mode': 'audio'}}],
            filter={'genres': ['pop'], 'instruments': ['viola']},
        )
        logs = []
        gen = planner._execute_plan(plan, {}, logs, target_song_count=50)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass
        assert calls == ['text_match']
        assert not any('backfill' in line for line in logs)

    def test_the_rerank_line_counts_songs_that_match_every_value(self):
        feats = {
            'both': {'mood_vector': 'pop:0.6', 'instrument_pct': {'viola': 0.99}},
            'pop_only': {'mood_vector': 'pop:0.6', 'instrument_pct': {'viola': 0.10}},
        }
        pool = [{'item_id': k, 'title': 'Song 1'} for k in feats]
        logs = []
        _final, matched, _moved = rerank.rerank(pool, {'genres': ['pop'], 'instruments': ['viola']}, feats, logs)
        assert matched == 1
        assert any('re-rank: 1/2 match the requested' in line and '1 match only some' in line for line in logs)

    def test_concept_percentiles_rank_against_the_library_sample(self, monkeypatch):
        import tasks.clap_steering as steering

        monkeypatch.setattr(calibration, 'concept_distributions', lambda: {'viola': np.array([0.1, 0.2, 0.3, 0.4])})
        monkeypatch.setattr(calibration, '_read_clap_embeddings', lambda ids: {i: np.zeros(512, np.float32) for i in ids})
        monkeypatch.setattr(steering, 'concept_scores', lambda m, terms: {'viola': np.array([0.35, 0.05])})
        out = calibration.concept_percentiles(['a', 'b'], ['viola', 'harp'])
        assert out == {'a': {'viola': 0.75}, 'b': {'viola': 0.0}}


class TestInstrumentalWording:
    def test_instrumental_passages_are_not_instrumental_tracks(self):
        assert 'instrumental' not in _hints('progressive rock with long instrumental passages')
        assert _hints('instrumental progressive rock')['instrumental'] is True

    def test_other_language_no_lyrics_keeps_the_instrumental_flag(self):
        plan = ToolPlan(filter={'instrumental': True})
        planner._strip_unrequested_filter_args(plan, {}, "Des chansons calmes, sans paroles", [])
        assert plan.filter == {'instrumental': True}
        assert _hints("Des chansons calmes pour m'endormir, sans paroles")['instrumental'] is True


class TestSeedModes:
    def _seed_call(self, seeds, blend='union'):
        return [{'name': 'seed_search', 'arguments': {'seeds': seeds, 'blend_mode': blend}}]

    def test_the_model_blend_choice_is_kept_for_two_seeds(self):
        seeds = [{'type': 'artist', 'name': 'Artist A'}, {'type': 'artist', 'name': 'Artist B'}]
        out = planner.validate_plan_args(
            self._seed_call(seeds), user_wants_rating=False,
            request_text='Artist A meets Artist B',
        )
        assert out[0]['arguments']['blend_mode'] == 'union'

    def test_from_one_seed_to_the_other_is_a_journey(self):
        seeds = [
            {'type': 'song', 'title': 'Song 1', 'artist': 'Artist A'},
            {'type': 'song', 'title': 'Song 2', 'artist': 'Artist B'},
        ]
        out = planner.validate_plan_args(
            self._seed_call(seeds), user_wants_rating=False,
            request_text='from Song 1 by Artist A to Song 2 by Artist B',
        )
        assert out[0]['arguments']['blend_mode'] == 'journey'

    def test_an_empty_subtract_does_not_hide_a_journey(self):
        seeds = [
            {'type': 'song', 'title': 'Song 1', 'artist': 'Artist A'},
            {'type': 'song', 'title': 'Song 2', 'artist': 'Artist B'},
        ]
        calls = [{'name': 'seed_search', 'arguments': {'seeds': seeds, 'blend_mode': 'subtract', 'subtract': []}}]
        out = planner.validate_plan_args(
            calls, user_wants_rating=False, request_text='A journey from Song 1 to Song 2',
        )
        assert out[0]['arguments']['blend_mode'] == 'journey'
        assert 'subtract' not in out[0]['arguments']

    def test_a_journey_needs_two_seeds(self):
        seeds = [{'type': 'artist', 'name': 'Artist A'}]
        out = planner.validate_plan_args(
            self._seed_call(seeds, 'journey'), user_wants_rating=False, request_text='a journey',
        )
        assert out[0]['arguments']['blend_mode'] == 'union'

    def test_a_journey_runs_alone_with_the_playlist_length(self):
        journey = {'name': 'seed_search', 'arguments': {'seeds': [], 'blend_mode': 'journey'}}
        plan = ToolPlan(primaries=[journey, {'name': 'text_match', 'arguments': {}}], filter={'genres': ['rock']})
        planner._prepare_journey(plan, 30, [])
        assert plan.primaries == [journey] and plan.filter is None
        assert journey['arguments']['journey_length'] == 30


class TestEnergyCalibration:
    def test_norm_maps_to_library_percentiles(self, monkeypatch):
        values = [0.5 + i * 0.02 for i in range(calibration.QUANTILE_STEPS + 1)]
        monkeypatch.setattr(calibration, 'energy_quantiles', lambda: values)
        assert calibration.energy_to_raw(0.0) == pytest.approx(0.5)
        assert calibration.energy_to_raw(0.5) == pytest.approx(0.7)
        assert calibration.energy_to_raw(1.0) == pytest.approx(0.9)
        assert calibration.energy_to_norm(0.7) == pytest.approx(0.5)
        assert calibration.energy_to_norm(0.1) == 0.0

    def test_falls_back_to_the_configured_scale(self, monkeypatch):
        monkeypatch.setattr(calibration, 'energy_quantiles', lambda: None)
        monkeypatch.setattr(config, 'ENERGY_MIN', 0.2)
        monkeypatch.setattr(config, 'ENERGY_MAX', 0.6)
        assert calibration.energy_to_raw(0.5) == pytest.approx(0.4)
        assert calibration.energy_to_norm(0.4) == pytest.approx(0.5)

    def test_a_failed_read_is_logged_and_cached_briefly(self, monkeypatch):
        calibration.reset_cache()
        monkeypatch.setattr(calibration, '_read_quantiles', Mock(side_effect=RuntimeError('db down')))
        try:
            assert calibration.energy_quantiles() is None
            assert calibration.energy_quantiles() is None
            assert calibration._read_quantiles.call_count == 1
        finally:
            calibration.reset_cache()


class TestInRangeTiers:
    def _order(self, filt, feats):
        pool = [{'item_id': k, 'title': f'Song {k}'} for k in feats]
        final, _matched, _moved = rerank.rerank(pool, filt, feats, [])
        return [s['item_id'] for s in final]

    def test_an_explicit_year_range_ranks_in_range_songs_first(self):
        feats = {'a': {'year': 1985}, 'b': {'year': 1994}, 'c': {'year': 2001}}
        assert self._order({'year_min': 1990, 'year_max': 1999}, feats)[0] == 'b'

    def test_an_exact_bpm_range_is_a_tier(self):
        feats = {'a': {'tempo': 150.0}, 'b': {'tempo': 171.0}, 'c': {'tempo': 120.0}}
        filt = {'tempo_min': 160, 'tempo_max': 180, rerank.EXACT_TEMPO_KEY: True}
        assert self._order(filt, feats) == ['b', 'a', 'c']

    def test_track_length_ranks_in_range_songs_first(self):
        feats = {'a': {'duration': 400.0}, 'b': {'duration': 150.0}}
        assert self._order({'duration_max': 180}, feats)[0] == 'b'

    def test_recently_added_ranks_new_songs_first(self):
        now = datetime.datetime.now()
        feats = {'a': {'created_at': now - datetime.timedelta(days=90)}, 'b': {'created_at': now}}
        assert self._order({'added_within_days': 30}, feats)[0] == 'b'

    def test_a_mid_range_never_scores_flat_outside_it(self):
        assert rerank._range_pref_score(0.9, 0.3, 0.5) > rerank._range_pref_score(1.0, 0.3, 0.5) > 0

    def test_a_one_sided_bound_is_a_gate_not_a_maximizer(self):
        assert rerank._range_pref_score(0.05, 0.0, 0.35) == rerank._range_pref_score(0.3, 0.0, 0.35) == 1.0
        assert rerank._range_pref_score(0.6, 0.0, 0.35) < 1.0

    def test_the_main_style_leads_among_genre_matches(self):
        feats = {
            'a': {'mood_vector': 'ambient:0.6,jazz:0.55,Progressive rock:0.51'},
            'b': {'mood_vector': 'Progressive rock:0.56,rock:0.53'},
        }
        assert self._order({'genres': ['Progressive rock']}, feats)[0] == 'b'

    def test_the_requested_genre_outranks_other_categorical_hits(self):
        feats = {
            'a': {'mood_vector': 'jazz:0.6,instrumental:0.6'},
            'b': {'mood_vector': 'Progressive rock:0.55'},
        }
        assert self._order({'genres': ['Progressive rock'], 'instrumental': True}, feats)[0] == 'b'

    def test_very_short_tracks_sink(self):
        feats = {'a': {'duration': 30.0, 'year': 1995}, 'b': {'duration': 200.0, 'year': 1995}}
        assert self._order({'year_min': 1990, 'year_max': 1999}, feats) == ['b', 'a']


def _cursor(rows):
    cur = MagicMock()
    cur.__enter__ = Mock(return_value=cur)
    cur.__exit__ = Mock(return_value=False)
    cur.fetchall = Mock(return_value=rows)
    return cur


class TestAlbumAndSql:
    def test_the_artist_name_is_not_taken_for_a_self_titled_album(self):
        cur = _cursor([('Artist A', 'Artist A', 10), ('Album Seventeen', 'Artist A', 12)])
        conn = MagicMock()
        conn.cursor = Mock(return_value=cur)
        with patch.object(tool_impl, 'get_db_connection', return_value=conn):
            hit = tool_impl._album_named_in_request('play the album Album Seventeen by Artist A', 'Artist A')
        assert hit['album'] == 'Album Seventeen'

    def test_duration_and_recently_added_bounds_reach_the_sql(self):
        cur = _cursor([])
        conn = MagicMock()
        conn.cursor = Mock(return_value=cur)
        with (
            patch.object(tool_impl, 'get_db_connection', return_value=conn),
            patch.object(tool_impl, '_server_availability_filter', return_value=('', [])),
        ):
            tool_impl._database_genre_query_sync(
                get_songs=10, duration_max=180, added_within_days=30
            )
        sql, params = cur.execute.call_args[0]
        assert 'duration <= %s' in sql and "created_at >= NOW()" in sql
        assert 180.0 in params and 30 in params
        assert sql.count('%s') == len(params)
