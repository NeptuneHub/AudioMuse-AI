# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""AI tool-plan normalization, vocabulary aliasing and prompt/schema shape.

Covers the planner that turns raw LLM tool arguments into a validated plan:
vocabulary remapping, mood/voice routing, argument validation, duplicate-call
dedupe/cap, and the derived prompt/grammar builders.

Main Features:
* Vocabulary aliases map short female/male tokens to voices and drop unknowns; exact canonical moods win over energy aliases
* Mood lists split out voices and energy phrases; non-canonical genres dropped with a note
* Plan args drop seedless searches, hallucinated min_rating and sub-1900 years; duplicate calls dropped and plans capped
* build_tool_calls_schema emits typed per-tool branches (reasoning first, name enum locked); prompts derive tool prose from the schemas
* Genre/negation hint extraction (incl. 4-digit decades), hint backstop, hallucinated year/instrumental/exclusion stripping (exclusions need a negation cue), whole-word artist-relax regex, similarity-blended re-rank with skit demotion and the instrumental dimension, exclusion hard cuts, empty/self-subtract coercion to union, underfilled-hard-filter broadening, and the zero-result replan
"""

import importlib
import re
import sys
import types


def _ensure_config_stub():
    if 'config' in sys.modules:
        return
    cfg = types.ModuleType('config')
    cfg.MOOD_LABELS = [
        'rock',
        'pop',
        'alternative',
        'indie',
        'electronic',
        'female vocalists',
        'dance',
        '00s',
        'alternative rock',
        'jazz',
        'beautiful',
        'metal',
        'chillout',
        'male vocalists',
        'classic rock',
        'soul',
        'indie rock',
        'Mellow',
        'electronica',
        '80s',
        'folk',
        '90s',
        'chill',
        'instrumental',
        'punk',
        'oldies',
        'blues',
        'hard rock',
        'ambient',
        'acoustic',
        'experimental',
        'female vocalist',
        'guitar',
        'Hip-Hop',
        '70s',
        'party',
        'country',
        'easy listening',
        'sexy',
        'catchy',
        'funk',
        'electro',
        'heavy metal',
        'Progressive rock',
        '60s',
        'rnb',
        'indie pop',
        'sad',
        'House',
        'happy',
    ]
    cfg.OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']
    cfg.VOICE_VOCAB = ['female vocalists', 'female vocalist', 'male vocalists']
    cfg.AI_FALLBACK_GENRES = 'rock, pop, jazz'
    cfg.AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX = 4
    cfg.AI_BRAINSTORM_SEED_ARTISTS_MAX = 4
    cfg.AI_BRAINSTORM_LYRIC_THEMES_MAX = 3
    cfg.AI_TOOLCALL_TEMPERATURE = 0.7
    cfg.CLAP_ENABLED = True
    cfg.LYRICS_ENABLED = True
    cfg.STRATIFIED_GENRES = [
        'rock',
        'pop',
        'alternative',
        'indie',
        'electronic',
        'jazz',
        'metal',
        'classic rock',
        'soul',
        'indie rock',
        'electronica',
        'folk',
        'punk',
        'blues',
        'hard rock',
        'ambient',
        'acoustic',
        'experimental',
        'Hip-Hop',
        'country',
        'funk',
        'electro',
        'heavy metal',
        'Progressive rock',
        'rnb',
        'indie pop',
        'House',
    ]
    sys.modules['config'] = cfg


def _vocab():
    _ensure_config_stub()
    import tasks.ai.vocab as v

    importlib.reload(v)
    return v


def _plan():
    _ensure_config_stub()
    import tasks.ai.planner as p

    importlib.reload(p)
    return p


class TestVocabAliases:
    def test_female_alone_maps_to_voices(self):
        v = _vocab()
        mv, of = v.normalize_mood('female')
        assert 'female vocalists' in mv
        assert 'female vocalist' in mv
        assert of == []

    def test_male_alone_maps_to_voices(self):
        v = _vocab()
        mv, of = v.normalize_mood('male')
        assert 'male vocalists' in mv
        assert of == []

    def test_female_voice_phrase_maps_to_voices(self):
        v = _vocab()
        mv, _ = v.normalize_mood('female voice')
        assert 'female vocalists' in mv
        assert 'female vocalist' in mv

    def test_unknown_short_token_dropped_not_remapped(self):
        v = _vocab()
        mv, of = v.normalize_mood('xz')
        assert mv == [] and of == []

    def test_remap_with_logging(self):
        v = _vocab()
        notes = []
        mv, of = v.normalize_mood('aggresive', notes=notes)
        assert 'aggressive' in of
        assert any("remapped mood" in n for n in notes)


class TestNormalizeMoodList:
    def test_voices_split_out_from_mood_vector(self):
        v = _vocab()
        result = v.normalize_mood_list(['female voice', 'happy'])
        assert 'female vocalists' in result['voices']
        assert 'female vocalist' in result['voices']
        assert 'happy' in result['other_features']
        assert result['mood_vector'] == []

    def test_energy_phrase_extracted(self):
        v = _vocab()
        result = v.normalize_mood_list(['chill'])
        assert abs(result['energy_min'] - 0.0) < 1e-9
        assert abs(result['energy_max'] - 0.35) < 1e-9

    def test_drop_notes_emitted_for_unknown(self):
        v = _vocab()
        result = v.normalize_mood_list(['totally-fake-mood-zzz'])
        assert any("dropped unrecognized mood" in n for n in result['notes'])


class TestNormalizeVoicesList:
    def test_voices_canonical_passthrough(self):
        v = _vocab()
        result = v.normalize_voices_list(['female vocalists', 'male vocalists'])
        assert result['voices'] == ['female vocalists', 'female vocalist', 'male vocalists']
        assert result['dropped'] == []

    def test_aliased_voices_normalized(self):
        v = _vocab()
        result = v.normalize_voices_list(['female voice', 'man singer'])
        assert 'female vocalists' in result['voices']
        assert 'male vocalists' in result['voices']

    def test_non_voice_dropped(self):
        v = _vocab()
        result = v.normalize_voices_list(['rock'])
        assert result['voices'] == []
        assert 'rock' in result['dropped']


class TestPlanFilterRouting:
    def test_moods_field_routed_to_voices_and_other_features(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [{'name': 'search_database', 'arguments': {'moods': ['female voice', 'happy']}}]
        )
        assert plan_obj.filter is not None
        assert 'female vocalists' in plan_obj.filter.get('voices', [])
        assert 'happy' in plan_obj.filter.get('moods', [])

    def test_voices_field_passes_through(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [{'name': 'search_database', 'arguments': {'voices': ['female voice']}}]
        )
        assert 'female vocalists' in plan_obj.filter['voices']

    def test_short_female_token_remapped(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [{'name': 'search_database', 'arguments': {'moods': ['female']}}]
        )
        assert 'female vocalists' in plan_obj.filter.get('voices', [])
        assert 'female vocalist' in plan_obj.filter.get('voices', [])

    def test_non_canonical_genre_dropped_with_note(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [{'name': 'search_database', 'arguments': {'genres': ['rock', 'definitelynotagenre']}}]
        )
        assert 'rock' in plan_obj.filter['genres']
        assert any("dropped" in n.lower() or "unrecognized" in n.lower() for n in plan_obj.notes)


class TestValidatePlanArgs:
    def test_drops_seed_search_with_no_usable_seeds(self):
        p = _plan()
        log = []
        out = p.validate_plan_args(
            [
                {
                    'name': 'seed_search',
                    'arguments': {'seeds': [{'type': 'song', 'title': '', 'song_artist': 'X'}]},
                }
            ],
            user_wants_rating=False,
            log_messages=log,
        )
        assert out == []

    def test_coerces_single_seed_alchemy_to_union(self):
        p = _plan()
        out = p.validate_plan_args(
            [
                {
                    'name': 'seed_search',
                    'arguments': {
                        'seeds': [{'type': 'artist', 'name': 'Madonna'}],
                        'blend_mode': 'alchemy',
                    },
                }
            ],
            user_wants_rating=False,
        )
        assert out[0]['name'] == 'seed_search'
        assert out[0]['arguments']['blend_mode'] == 'union'
        assert out[0]['arguments']['seeds'][0]['name'] == 'Madonna'

    def test_coerces_empty_subtract_to_union(self):
        p = _plan()
        log = []
        out = p.validate_plan_args(
            [
                {
                    'name': 'seed_search',
                    'arguments': {
                        'seeds': [
                            {'type': 'artist', 'name': 'Wu-Tang Clan'},
                            {'type': 'artist', 'name': 'Nujabes'},
                        ],
                        'blend_mode': 'subtract',
                        'subtract': [],
                    },
                }
            ],
            user_wants_rating=False,
            log_messages=log,
        )
        assert len(out) == 1
        assert out[0]['arguments']['blend_mode'] == 'union'
        assert 'subtract' not in out[0]['arguments']

    def test_coerces_self_subtraction_to_union(self):
        p = _plan()
        log = []
        seeds = [
            {'type': 'artist', 'name': 'Wu-Tang Clan'},
            {'type': 'artist', 'name': 'Nujabes'},
        ]
        out = p.validate_plan_args(
            [
                {
                    'name': 'seed_search',
                    'arguments': {
                        'seeds': list(seeds),
                        'blend_mode': 'subtract',
                        'subtract': [dict(s) for s in seeds],
                    },
                }
            ],
            user_wants_rating=False,
            log_messages=log,
        )
        assert len(out) == 1
        assert out[0]['arguments']['blend_mode'] == 'union'
        assert 'subtract' not in out[0]['arguments']
        assert any('self-subtraction' in ln for ln in log)

    def test_partial_self_subtraction_keeps_other_items(self):
        p = _plan()
        out = p.validate_plan_args(
            [
                {
                    'name': 'seed_search',
                    'arguments': {
                        'seeds': [{'type': 'artist', 'name': 'Oasis'}],
                        'blend_mode': 'subtract',
                        'subtract': [
                            {'type': 'artist', 'name': 'Oasis'},
                            {'type': 'artist', 'name': 'Blur'},
                        ],
                    },
                }
            ],
            user_wants_rating=False,
        )
        assert out[0]['arguments']['blend_mode'] == 'subtract'
        assert out[0]['arguments']['subtract'] == [{'type': 'artist', 'name': 'Blur'}]

    def test_strips_hallucinated_min_rating(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'search_database', 'arguments': {'genres': ['rock'], 'min_rating': 4}}],
            user_wants_rating=False,
        )
        assert 'min_rating' not in out[0]['arguments']

    def test_preserves_min_rating_when_user_asked(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'search_database', 'arguments': {'min_rating': 4}}],
            user_wants_rating=True,
        )
        assert out[0]['arguments']['min_rating'] == 4

    def test_strips_year_below_1900(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'search_database', 'arguments': {'genres': ['rock'], 'year_min': 1}}],
            user_wants_rating=False,
        )
        assert 'year_min' not in out[0]['arguments']

    def test_drops_filterless_search_database(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'search_database', 'arguments': {}}],
            user_wants_rating=False,
        )
        assert out == []


class TestMultiIntentClassification:
    def test_two_primaries_preserved(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [
                {
                    'name': 'seed_search',
                    'arguments': {'seeds': [{'type': 'artist', 'name': 'blink-182'}]},
                },
                {
                    'name': 'seed_search',
                    'arguments': {'seeds': [{'type': 'artist', 'name': 'Green Day'}]},
                },
            ]
        )
        assert len(plan_obj.primaries) == 2
        assert plan_obj.filter is None

    def test_primaries_plus_filter(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan(
            [
                {
                    'name': 'seed_search',
                    'arguments': {
                        'seeds': [
                            {
                                'type': 'song',
                                'title': 'By The Way',
                                'artist': 'Red Hot Chili Peppers',
                            }
                        ]
                    },
                },
                {'name': 'search_database', 'arguments': {'voices': ['female voice']}},
            ]
        )
        assert len(plan_obj.primaries) == 1
        assert plan_obj.filter is not None
        assert 'female vocalists' in plan_obj.filter['voices']


class TestIntentPreextract:
    def test_year_detected(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints

        h = extract_hints("2026 songs")
        assert h['year_min'] == 2026
        assert h['year_max'] == 2026

    def test_decade_detected(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints

        h = extract_hints("90s pop")
        assert h['year_min'] == 1990
        assert h['year_max'] == 1999

    def test_four_digit_decade_detected(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints

        h = extract_hints("chill jazzy hip hop from the 2000s")
        assert h['year_min'] == 2000
        assert h['year_max'] == 2009
        h = extract_hints("best of the 1970s")
        assert h['year_min'] == 1970
        assert h['year_max'] == 1979

    def test_energy_phrase_detected(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints

        h = extract_hints("chill jazz")
        assert abs(h['energy_min'] - 0.0) < 1e-9
        assert abs(h['energy_max'] - 0.35) < 1e-9

    def test_explicit_energy_floor(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints

        h = extract_hints("songs with energy above 0.5")
        assert h['energy_min'] == 0.5


def _prompts():
    _ensure_config_stub()
    import tasks.ai.prompts as pr

    importlib.reload(pr)
    return pr


def _tools_fixture():
    return [
        {
            'name': 'seed_search',
            'description': 'Find songs similar to seed songs or artists.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'seeds': {'type': 'array', 'items': {'type': 'object'}},
                    'blend_mode': {'type': 'string', 'enum': ['union', 'alchemy', 'subtract']},
                    'subtract': {'type': 'array'},
                },
                'required': ['seeds'],
            },
        },
        {
            'name': 'text_match',
            'description': 'Find songs from a free-text description.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'mode': {'type': 'string', 'enum': ['audio', 'lyrics']},
                },
                'required': ['query'],
            },
        },
        {
            'name': 'knowledge_lookup',
            'description': 'Answer popularity or cultural requests.',
            'inputSchema': {
                'type': 'object',
                'properties': {'user_request': {'type': 'string'}},
                'required': ['user_request'],
            },
        },
        {
            'name': 'search_database',
            'description': 'Filter the library by exact metadata.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'genres': {'type': 'array'},
                    'artist': {'type': 'string'},
                    'year_min': {'type': 'integer'},
                },
            },
        },
    ]


class TestDedupeAndCapCalls:
    def test_duplicates_dropped_order_preserved(self):
        p = _plan()
        log = []
        calls = [
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'A'}]}},
            {'name': 'search_database', 'arguments': {'genres': ['rock']}},
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'A'}]}},
        ]
        out = p.dedupe_and_cap_calls(calls, log_messages=log)
        assert [c['name'] for c in out] == ['seed_search', 'search_database']
        assert any('duplicate' in ln for ln in log)

    def test_same_tool_different_args_kept(self):
        p = _plan()
        calls = [
            {'name': 'search_database', 'arguments': {'genres': ['rock']}},
            {'name': 'search_database', 'arguments': {'genres': ['jazz']}},
        ]
        out = p.dedupe_and_cap_calls(calls)
        assert len(out) == 2

    def test_cap_at_max(self):
        p = _plan()
        calls = [
            {'name': 'search_database', 'arguments': {'year_min': 1990 + i}} for i in range(6)
        ]
        log = []
        out = p.dedupe_and_cap_calls(calls, log_messages=log)
        assert len(out) == p.MAX_TOOL_CALLS
        assert any('capping' in ln for ln in log)

    def test_non_dict_entries_skipped(self):
        p = _plan()
        out = p.dedupe_and_cap_calls(['bogus', {'name': 'text_match', 'arguments': {}}])
        assert len(out) == 1


class TestToolCallsSchema:
    def test_reasoning_first_and_required(self):
        pr = _prompts()
        schema = pr.build_tool_calls_schema(_tools_fixture())
        assert next(iter(schema['properties'])) == 'reasoning'
        assert schema['required'] == ['reasoning', 'tool_calls']

    def test_branches_lock_name_and_type_arguments(self):
        pr = _prompts()
        schema = pr.build_tool_calls_schema(_tools_fixture())
        branches = schema['properties']['tool_calls']['items']['oneOf']
        assert len(branches) == 4
        by_name = {b['properties']['name']['enum'][0]: b for b in branches}
        assert set(by_name) == {'seed_search', 'text_match', 'knowledge_lookup', 'search_database'}
        sd_args = by_name['search_database']['properties']['arguments']
        assert sd_args['additionalProperties'] is False
        assert 'artist' in sd_args['properties']
        seed_args = by_name['seed_search']['properties']['arguments']
        assert seed_args['required'] == ['seeds']

    def test_source_schema_not_mutated(self):
        pr = _prompts()
        tools = _tools_fixture()
        pr.build_tool_calls_schema(tools)
        assert 'additionalProperties' not in tools[0]['inputSchema']

    def test_derived_schema_carries_no_unique_items_and_relies_on_max_items(self):
        pr = _prompts()
        tools = _tools_fixture()

        schema = pr.build_tool_calls_schema(tools)
        branches = schema['properties']['tool_calls']['items']['oneOf']
        argument_schemas = [branch['properties']['arguments'] for branch in branches]

        def array_schemas(value):
            if isinstance(value, dict):
                if value.get('type') == 'array':
                    yield value
                for child in value.values():
                    yield from array_schemas(child)
            elif isinstance(value, list):
                for child in value:
                    yield from array_schemas(child)

        derived_arrays = [
            array_schema
            for argument_schema in argument_schemas
            for array_schema in array_schemas(argument_schema)
        ]

        assert derived_arrays
        assert all('uniqueItems' not in array_schema for array_schema in derived_arrays)

    def test_derived_argument_arrays_preserve_the_source_max_items_cap(self):
        pr = _prompts()
        tools = _tools_fixture()
        for tool in tools:
            if tool['name'] == 'search_database':
                tool['inputSchema']['properties']['genres']['maxItems'] = 5

        schema = pr.build_tool_calls_schema(tools)
        branch = next(
            b for b in schema['properties']['tool_calls']['items']['oneOf']
            if b['properties']['name']['enum'] == ['search_database']
        )
        assert branch['properties']['arguments']['properties']['genres']['maxItems'] == 5

    def test_tool_calls_schema_max_items_matches_the_planner_cap(self):
        pr = _prompts()
        planner = _plan()
        schema = pr.build_tool_calls_schema(_tools_fixture())
        assert schema['properties']['tool_calls']['maxItems'] == planner.MAX_TOOL_CALLS


class TestPromptRendering:
    def test_system_prompt_derives_tools_and_rules(self):
        pr = _prompts()
        text = pr.build_mcp_system_prompt(_tools_fixture())
        for name in ('seed_search', 'text_match', 'knowledge_lookup', 'search_database'):
            assert name in text
        assert "artist='X'" in text
        assert 'search_database(genres, artist, year_min)' in text

    def test_ollama_prompt_contract_and_examples(self):
        pr = _prompts()
        text = pr.build_ollama_tool_calling_prompt('play jazz', _tools_fixture())
        assert '"reasoning"' in text
        assert '"tool_calls"' in text
        assert text.count('"name"') >= 4
        assert 'play jazz' in text
        assert 'Johnny Cash' in text

    def test_examples_shrink_with_tool_surface(self):
        pr = _prompts()
        only_filter = [t for t in _tools_fixture() if t['name'] == 'search_database']
        text = pr.build_ollama_tool_calling_prompt('x', only_filter)
        assert 'Daft Punk' not in text
        assert 'seed_search' not in text


class TestVocabExactMoodPrecedence:
    def test_relaxed_stays_a_mood(self):
        v = _vocab()
        result = v.normalize_mood_list(['relaxed'])
        assert result['other_features'] == ['relaxed']
        assert result['energy_min'] is None
        assert result['energy_max'] is None

    def test_chill_still_maps_to_energy(self):
        v = _vocab()
        result = v.normalize_mood_list(['chill'])
        assert abs(result['energy_min'] - 0.0) < 1e-9
        assert abs(result['energy_max'] - 0.35) < 1e-9


class TestGenreAndNegationHints:
    def test_negated_genre_detected(self):
        p = _plan()
        h = p.extract_hints("upbeat party songs but absolutely no rap")
        assert h.get('exclude_genres') == ['Hip-Hop']
        assert 'genres' not in h

    def test_positive_genre_detected(self):
        p = _plan()
        h = p.extract_hints("pop songs that talk about love")
        assert h.get('genres') == ['pop']

    def test_positive_and_negated_split(self):
        p = _plan()
        h = p.extract_hints("rock songs but not metal")
        assert h.get('genres') == ['rock']
        assert h.get('exclude_genres') == ['metal']

    def test_duration_flagged_unsupported(self):
        p = _plan()
        h = p.extract_hints("short punchy songs under 3 minutes")
        assert any('duration' in u for u in h.get('unsupported', []))


class TestHintBackstop:
    def test_missing_genre_merged(self):
        p = _plan()
        plan = p.ToolPlan(
            primaries=[{'name': 'text_match', 'arguments': {'query': 'love', 'mode': 'lyrics'}}]
        )
        p._apply_hint_backstop(plan, {'genres': ['pop']}, [])
        assert plan.filter['genres'] == ['pop']

    def test_bpm_becomes_tempo_window(self):
        p = _plan()
        plan = p.ToolPlan(
            primaries=[{'name': 'text_match', 'arguments': {'query': 'x', 'mode': 'audio'}}]
        )
        p._apply_hint_backstop(plan, {'bpm': 170}, [])
        assert abs(plan.filter['tempo_min'] - 160.0) < 1e-9
        assert abs(plan.filter['tempo_max'] - 180.0) < 1e-9

    def test_model_args_not_overridden(self):
        p = _plan()
        plan = p.ToolPlan(filter={'tempo_min': 100.0})
        p._apply_hint_backstop(plan, {'bpm': 170}, [])
        assert abs(plan.filter['tempo_min'] - 100.0) < 1e-9
        assert 'tempo_max' not in plan.filter


class TestStripUnrequestedArgs:
    def test_year_stripped_without_year_in_request(self):
        p = _plan()
        plan = p.ToolPlan(filter={'year_min': 1900, 'year_max': 2100, 'genres': ['rock']})
        p._strip_unrequested_filter_args(plan, {}, "energetic rock for the gym", [])
        assert 'year_min' not in plan.filter
        assert 'year_max' not in plan.filter
        assert plan.filter['genres'] == ['rock']

    def test_year_kept_for_yearish_word(self):
        p = _plan()
        plan = p.ToolPlan(filter={'year_min': 2015, 'year_max': 2024, 'genres': ['rock']})
        p._strip_unrequested_filter_args(plan, {}, "recent rock hits", [])
        assert plan.filter['year_min'] == 2015

    def test_instrumental_false_stripped(self):
        p = _plan()
        plan = p.ToolPlan(filter={'instrumental': False, 'genres': ['rock']})
        p._strip_unrequested_filter_args(plan, {}, "punchy rock", [])
        assert 'instrumental' not in plan.filter

    def test_instrumental_kept_when_requested(self):
        p = _plan()
        plan = p.ToolPlan(filter={'instrumental': True})
        p._strip_unrequested_filter_args(plan, {'instrumental': True}, "instrumental rock", [])
        assert plan.filter['instrumental'] is True

    def test_exclusions_stripped_without_negation_in_request(self):
        p = _plan()
        plan = p.ToolPlan(
            filter={
                'artist': 'Emmylou Harris',
                'genres': ['country', 'folk'],
                'exclude_genres': ['Hip-Hop', 'rock', 'pop'],
                'exclude_artists': ['Pitbull'],
            }
        )
        log = []
        p._strip_unrequested_filter_args(
            plan,
            {'genres': ['country', 'folk']},
            "Emmylou Harris songs plus similar country folk artists, only from the 70s",
            log,
        )
        assert 'exclude_genres' not in plan.filter
        assert 'exclude_artists' not in plan.filter
        assert plan.filter['artist'] == 'Emmylou Harris'
        assert any('hallucinated exclusions' in ln for ln in log)

    def test_exclusions_kept_when_request_negates(self):
        p = _plan()
        plan = p.ToolPlan(
            filter={
                'moods': ['party'],
                'exclude_genres': ['Hip-Hop'],
                'exclude_artists': ['Pitbull'],
            }
        )
        p._strip_unrequested_filter_args(
            plan,
            {'exclude_genres': ['Hip-Hop']},
            "upbeat party songs but absolutely no rap and nothing by Pitbull",
            [],
        )
        assert plan.filter['exclude_genres'] == ['Hip-Hop']
        assert plan.filter['exclude_artists'] == ['Pitbull']

    def test_artist_exclusion_kept_on_negation_cue_alone(self):
        p = _plan()
        plan = p.ToolPlan(filter={'genres': ['Hip-Hop'], 'exclude_artists': ['50 Cent']})
        p._strip_unrequested_filter_args(
            plan, {}, "Hip hop songs but absolutely no 50 Cent", []
        )
        assert plan.filter['exclude_artists'] == ['50 Cent']


class TestArtistWordRegex:
    def test_word_boundary_pattern(self):
        _ensure_config_stub()
        from tasks.ai.tool_impl import _artist_word_regex

        assert _artist_word_regex('Nas') == r'\mNas\M'
        assert _artist_word_regex('  AC/DC ') == r'\mAC/DC\M'


class TestRerankSimilarityBlend:
    def _pool(self):
        songs = [
            {'item_id': 'a', 'title': 'Song A'},
            {'item_id': 'b', 'title': 'Song B'},
            {'item_id': 'c', 'title': 'Song C'},
        ]
        feats = {
            'a': {'other_features': 'party:0.50'},
            'b': {'other_features': 'party:0.52'},
            'c': {'other_features': 'party:0.60'},
        }
        return songs, feats

    def test_without_sim_pure_filter_order(self):
        from tasks.ai import rerank
        songs, feats = self._pool()
        final, matched, _moved = rerank.rerank(songs, {'moods': ['party']}, feats, [])
        assert [s['item_id'] for s in final] == ['c', 'b', 'a']
        assert matched == 3

    def test_sim_rank_blended_into_order(self):
        from tasks.ai import rerank
        songs, feats = self._pool()
        final, _matched, _moved = rerank.rerank(
            songs,
            {'moods': ['party']},
            feats,
            [],
            sim_by_id={'a': 1.0, 'b': 0.0, 'c': 0.9},
        )
        assert [s['item_id'] for s in final] == ['c', 'a', 'b']

    def test_skit_title_demoted_to_end(self):
        from tasks.ai import rerank
        songs = [
            {'item_id': 'a', 'title': 'Party Anthem'},
            {'item_id': 'b', 'title': 'Party Interlude'},
            {'item_id': 'c', 'title': 'Dance Night'},
        ]
        feats = {
            'a': {'other_features': 'party:0.50'},
            'b': {'other_features': 'party:0.90'},
            'c': {'other_features': 'party:0.60'},
        }
        final, _matched, _moved = rerank.rerank(songs, {'moods': ['party']}, feats, [])
        assert [s['item_id'] for s in final] == ['c', 'a', 'b']


class TestInstrumentalRerank:
    def test_dim_scores_instrumental_true(self):
        from tasks.ai import rerank
        s = rerank._filter_dim_scores(
            {'instrumental': True}, {'mood_vector': 'instrumental:0.62,jazz:0.30'}
        )
        assert abs(s['instrumental'] - 0.62) < 1e-9
        s = rerank._filter_dim_scores({'instrumental': True}, {'mood_vector': 'pop:0.50'})
        assert abs(s['instrumental'] - 0.0) < 1e-9

    def test_dim_scores_instrumental_false(self):
        from tasks.ai import rerank
        s = rerank._filter_dim_scores(
            {'instrumental': False}, {'mood_vector': 'instrumental:0.80'}
        )
        assert abs(s['instrumental'] - 0.2) < 1e-9

    def test_instrumental_tracks_rank_first(self):
        from tasks.ai import rerank
        songs = [
            {'item_id': 'v', 'title': 'Vocal Hit'},
            {'item_id': 'i', 'title': 'Guitar Study'},
        ]
        feats = {
            'v': {'mood_vector': 'pop:0.90'},
            'i': {'mood_vector': 'instrumental:0.60'},
        }
        final, matched, _moved = rerank.rerank(
            songs, {'instrumental': True}, feats, [], sim_by_id={'v': 1.0, 'i': 0.5}
        )
        assert [s['item_id'] for s in final] == ['i', 'v']
        assert matched == 1


class TestUnderfilledBroadening:
    def _run(self, monkeypatch, strict_songs, broad_songs, feats, target=3):
        p = _plan()
        import tasks.ai.tools as tools_mod
        import tasks.ai.tool_impl as impl_mod

        seen_args = []

        def fake_exec(name, args, cfg):
            seen_args.append(args)
            if args.get('moods'):
                return {'songs': list(strict_songs), 'message': 'strict'}
            return {'songs': list(broad_songs), 'message': 'broad'}

        monkeypatch.setattr(tools_mod, 'execute_mcp_tool', fake_exec)
        monkeypatch.setattr(impl_mod, '_fetch_pool_features', lambda ids: feats)

        logs = []
        result = p._run_search_database_with_relax(
            {'moods': ['relaxed'], 'get_songs': 200},
            {'provider': 'NONE'},
            target,
            logs,
            pool_target=1000,
        )
        return result, logs, seen_args

    def test_underfilled_hard_cut_broadens_and_soft_reranks(self, monkeypatch):
        strict = [{'item_id': 'r2', 'title': 'R2', 'artist': 'B'}]
        broad = [
            {'item_id': 'r1', 'title': 'R1', 'artist': 'A'},
            {'item_id': 'r2', 'title': 'R2', 'artist': 'B'},
            {'item_id': 'r3', 'title': 'R3', 'artist': 'C'},
        ]
        feats = {
            'r1': {'other_features': 'relaxed:0.20'},
            'r2': {'other_features': 'relaxed:0.90'},
            'r3': {'other_features': 'relaxed:0.50'},
        }
        result, logs, seen_args = self._run(monkeypatch, strict, broad, feats)
        assert [s['item_id'] for s in result['songs']] == ['r2', 'r3', 'r1']
        assert seen_args[-1].get('get_songs') == 1000
        assert 'moods' not in seen_args[-1]
        assert any('underfilled' in ln for ln in logs)

    def test_filled_hard_cut_not_broadened(self, monkeypatch):
        strict = [
            {'item_id': 's1', 'title': 'S1', 'artist': 'A'},
            {'item_id': 's2', 'title': 'S2', 'artist': 'B'},
            {'item_id': 's3', 'title': 'S3', 'artist': 'C'},
        ]
        result, logs, seen_args = self._run(monkeypatch, strict, [], {}, target=3)
        assert [s['item_id'] for s in result['songs']] == ['s1', 's2', 's3']
        assert all(a.get('moods') for a in seen_args)
        assert not any('underfilled' in ln for ln in logs)


class TestExclusionsHardCut:
    def test_exclude_artist_and_genre(self):
        p = _plan()
        songs = [
            {'item_id': '1', 'title': 'S1', 'artist': '50 Cent'},
            {'item_id': '2', 'title': 'S2', 'artist': 'Enya'},
            {'item_id': '3', 'title': 'S3', 'artist': 'Mobb Deep'},
        ]
        feats = {
            '1': {'author': '50 Cent', 'mood_vector': ''},
            '2': {'author': 'Enya', 'mood_vector': 'pop:0.1'},
            '3': {'author': 'Mobb Deep', 'mood_vector': 'Hip-Hop:0.8'},
        }
        kept = p._apply_exclusions(
            songs,
            {'exclude_artists': ['50 cent'], 'exclude_genres': ['Hip-Hop']},
            feats,
            [],
        )
        assert [s['item_id'] for s in kept] == ['2']

    def test_no_exclusions_is_noop(self):
        p = _plan()
        songs = [{'item_id': '1', 'title': 'S1', 'artist': 'X'}]
        assert p._apply_exclusions(songs, {'genres': ['rock']}, {}, []) == songs


class TestPlanNormalizationExclusions:
    def test_exclusion_args_survive_and_normalize(self):
        p = _plan()
        plan = p.validate_and_normalize_plan(
            [
                {
                    'name': 'search_database',
                    'arguments': {
                        'exclude_genres': ['rap'],
                        'exclude_artists': ['50 Cent', '50 Cent', ''],
                    },
                }
            ]
        )
        assert plan.filter['exclude_genres'] == ['Hip-Hop']
        assert plan.filter['exclude_artists'] == ['50 Cent']

    def test_exclusion_only_filter_counts_as_content(self):
        p = _plan()
        assert p._has_filter_content({'exclude_artists': ['X']})

    def test_conflicting_genre_exclusion_wins(self):
        p = _plan()
        plan = p.validate_and_normalize_plan(
            [
                {
                    'name': 'search_database',
                    'arguments': {'genres': ['Hip-Hop'], 'exclude_genres': ['rap']},
                }
            ]
        )
        assert 'genres' not in plan.filter
        assert plan.filter['exclude_genres'] == ['Hip-Hop']


class TestReasoningSchemaCap:
    def test_reasoning_has_maxlength(self):
        pr = _prompts()
        schema = pr.build_tool_calls_schema(_tools_fixture())
        assert schema['properties']['reasoning']['maxLength'] == 300


class TestZeroResultReplan:
    def test_replans_once_with_feedback(self, monkeypatch):
        p = _plan()
        calls = []

        def fake_ai(user_message, tools, ai_config, log_messages, library_context=None):
            calls.append(user_message)
            if len(calls) == 1:
                return {
                    'tool_calls': [
                        {'name': 'search_database', 'arguments': {'artist': 'Nobody'}}
                    ]
                }
            return {
                'tool_calls': [{'name': 'search_database', 'arguments': {'genres': ['rock']}}]
            }

        monkeypatch.setattr(p, 'call_ai_for_plan', fake_ai)

        import tasks.ai.tools as tools_mod

        def fake_exec(name, args, cfg):
            if args.get('artist') == 'Nobody':
                return {'songs': [], 'message': 'Found 0 songs matching artist: Nobody'}
            return {'songs': [{'item_id': 'x1', 'title': 'T', 'artist': 'A'}], 'message': 'ok'}

        monkeypatch.setattr(tools_mod, 'execute_mcp_tool', fake_exec)

        logs = []
        gen = p.plan_and_execute_once('songs by Nobody', [], {'provider': 'NONE'}, logs)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            result = stop.value

        assert result['songs']
        assert len(calls) == 2
        assert 'PREVIOUS ATTEMPT FAILED' in calls[1]


def _drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _example_calls(example_text):
    import json

    return json.loads(example_text.split('\n', 1)[1])['tool_calls']


def _run_plan(p, monkeypatch, request, tool_calls, raw_request=None):
    import tasks.ai.tools as tools_mod
    import tasks.ai.tool_impl as impl_mod

    seen = []

    def fake_ai(user_message, tools, ai_config, log_messages, library_context=None):
        if callable(tool_calls):
            return tool_calls(user_message)
        return {'tool_calls': [dict(c) for c in tool_calls]}

    def fake_exec(name, args, cfg):
        seen.append((name, {k: v for k, v in args.items() if k != 'get_songs'}))
        return {'songs': [{'item_id': 'x1', 'title': 'T', 'artist': 'A'}], 'message': 'ok'}

    monkeypatch.setattr(p, 'call_ai_for_plan', fake_ai)
    monkeypatch.setattr(tools_mod, 'execute_mcp_tool', fake_exec)
    monkeypatch.setattr(
        impl_mod, '_fetch_pool_features',
        lambda ids: {i: {'author': 'A', 'mood_vector': '', 'other_features': ''} for i in ids},
    )
    logs = []
    result = _drive(p.plan_and_execute_once(
        request, [], {'provider': 'NONE'}, logs, raw_user_request=raw_request,
    ))
    return result, seen, logs


class TestListArgDedupe:
    def test_repeated_exclude_genres_collapse_to_one_value(self):
        p = _plan()
        calls = [{'name': 'search_database',
                  'arguments': {'exclude_genres': ['Hip-Hop'] * 5}}]
        logs = []
        p._dedupe_call_lists(calls, log_messages=logs)
        assert calls[0]['arguments']['exclude_genres'] == ['Hip-Hop']
        assert any('dedupe:' in line for line in logs)

    def test_dedupe_is_case_insensitive_and_keeps_first_seen_order(self):
        p = _plan()
        calls = [{'name': 'search_database',
                  'arguments': {'genres': ['rock', 'ROCK', 'jazz', 'Rock']}}]
        p._dedupe_call_lists(calls, log_messages=[])
        assert calls[0]['arguments']['genres'] == ['rock', 'jazz']

    def test_repeated_seed_search_seeds_collapse_by_identity(self):
        p = _plan()
        calls = [{'name': 'seed_search', 'arguments': {'seeds': [
            {'type': 'artist', 'name': 'Oasis'},
            {'type': 'artist', 'name': 'oasis'},
            {'type': 'artist', 'name': 'Blur'},
        ]}}]
        p._dedupe_call_lists(calls, log_messages=[])
        assert calls[0]['arguments']['seeds'] == [
            {'type': 'artist', 'name': 'Oasis'},
            {'type': 'artist', 'name': 'Blur'},
        ]

    def test_repeated_subtract_items_collapse_by_identity(self):
        p = _plan()
        calls = [{'name': 'seed_search', 'arguments': {
            'seeds': [{'type': 'artist', 'name': 'Oasis'}],
            'subtract': [{'type': 'artist', 'name': 'Blur'}] * 3,
        }}]
        p._dedupe_call_lists(calls, log_messages=[])
        assert calls[0]['arguments']['subtract'] == [{'type': 'artist', 'name': 'Blur'}]

    def test_calls_differing_only_by_padding_collapse_and_free_a_cap_slot(self):
        p = _plan()
        calls = [
            {'name': 'search_database', 'arguments': {'exclude_genres': ['pop', 'pop']}},
            {'name': 'search_database', 'arguments': {'exclude_genres': ['pop']}},
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'A'}]}},
            {'name': 'text_match', 'arguments': {'query': 'warm', 'mode': 'audio'}},
            {'name': 'knowledge_lookup', 'arguments': {'user_request': 'best of'}},
        ]
        deduped = p.dedupe_and_cap_calls(
            p._dedupe_call_lists(calls, log_messages=[]), log_messages=[]
        )
        assert [c['name'] for c in deduped] == [
            'search_database', 'seed_search', 'text_match', 'knowledge_lookup',
        ]


class TestContradictoryExclusionStrip:
    def test_exclude_artist_matching_a_seed_is_dropped(self):
        p = _plan()
        plan = p.ToolPlan(
            primaries=[{
                'name': 'seed_search',
                'arguments': {'seeds': [{'type': 'artist', 'name': 'Miles Davis'}]},
            }],
            filter={'exclude_artists': ['Miles Davis'], 'year_max': 1970},
        )

        p._strip_contradictory_exclusions(
            plan, {}, 'like Miles Davis, nothing after 1970', []
        )

        assert not (plan.filter or {}).get('exclude_artists')
        assert plan.filter['year_max'] == 1970

    def test_exclude_artist_matching_the_filter_artist_is_dropped(self):
        p = _plan()
        plan = p.ToolPlan(
            filter={'artist': 'Nas', 'exclude_artists': ['Nas'], 'album': 'Illmatic'}
        )
        p._strip_contradictory_exclusions(plan, {}, 'play the album Illmatic by Nas', [])
        assert not (plan.filter or {}).get('exclude_artists')

    def test_exclude_genre_the_user_asked_for_positively_is_dropped(self):
        p = _plan()
        plan = p.ToolPlan(filter={'genres': ['jazz'], 'exclude_genres': ['jazz']})
        p._strip_contradictory_exclusions(
            plan, {'genres': ['jazz'], 'exclude_genres': []},
            'relaxing jazz, nothing loud', [],
        )
        assert not (plan.filter or {}).get('exclude_genres')
        assert plan.filter['genres'] == ['jazz']

    def test_exclude_genre_absent_from_message_and_negated_hints_is_dropped(self):
        p = _plan()
        plan = p.ToolPlan(filter={'exclude_genres': ['classic rock', 'jazz']})
        p._strip_contradictory_exclusions(
            plan, {'genres': [], 'exclude_genres': ['jazz']},
            'chill music for studying, but no classical and no jazz', [],
        )
        assert plan.filter['exclude_genres'] == ['jazz']

    def test_exclude_genre_named_in_the_message_survives_when_the_hint_regex_missed_it(self):
        p = _plan()
        plan = p.ToolPlan(filter={'exclude_genres': ['country']})
        p._strip_contradictory_exclusions(
            plan, {'genres': [], 'exclude_genres': ['pop']},
            'no pop and no country please', [],
        )
        assert plan.filter['exclude_genres'] == ['country']

    def test_misspelled_artist_exclusion_survives_via_fuzzy_message_match(self):
        p = _plan()
        plan = p.ToolPlan(filter={'exclude_artists': ['Bee Gees']})
        p._strip_contradictory_exclusions(
            plan, {}, 'disco hits but absolutely nothing by the Beegees', [],
        )
        assert plan.filter['exclude_artists'] == ['Bee Gees']

    def test_exclude_artist_named_nowhere_in_the_request_is_dropped(self):
        p = _plan()
        plan = p.ToolPlan(filter={'genres': ['pop'], 'exclude_artists': ['Nickelback']})
        p._strip_contradictory_exclusions(plan, {}, 'upbeat pop, nothing slow', [])
        assert not (plan.filter or {}).get('exclude_artists')

    def test_hate_cue_marks_a_genre_as_negated_not_positive(self):
        p = _plan()
        hints = p.extract_hints('party music, I hate country')
        assert hints.get('exclude_genres') == ['country']
        assert 'country' not in [g.lower() for g in hints.get('genres') or []]

    def test_exclusions_that_would_empty_the_pool_are_reverted_with_a_note(self):
        p = _plan()
        pool = [{'item_id': 'a', 'title': 'T', 'artist': 'Miles Davis'}]
        feats = {'a': {'author': 'Miles Davis', 'mood_vector': ''}}
        notes = []
        kept = p._apply_exclusions(
            pool, {'exclude_artists': ['Miles Davis']}, feats, [], notes=notes
        )
        assert kept == pool
        assert notes


class TestEmptyPlanRescue:
    def test_empty_search_database_never_returns_an_error_key(self, monkeypatch):
        p = _plan()
        result, seen, _logs = _run_plan(
            p, monkeypatch, 'Build a playlist for: "music"',
            [{'name': 'search_database', 'arguments': {}}], raw_request='music',
        )
        assert 'error' not in result
        assert result['songs']
        assert ('text_match', {'query': 'music', 'mode': 'audio'}) in seen

    def test_rescue_uses_the_raw_user_words_not_the_wrapper(self, monkeypatch):
        p = _plan()
        _result, seen, _logs = _run_plan(
            p, monkeypatch, 'Build a 100-song playlist for: "calm piano"',
            [], raw_request='calm piano',
        )
        queries = [a.get('query') for n, a in seen if n == 'text_match']
        assert queries == ['calm piano']

    def test_rescue_attaches_extracted_hints_as_the_plan_filter(self):
        p = _plan()
        request = 'fast 170 bpm rock from the 90s'
        plan = p._synthesize_rescue_plan(request, p.extract_hints(request), [])
        assert plan.filter['year_min'] == 1990
        assert plan.filter['year_max'] == 1999
        assert plan.filter['genres'] == ['rock']

    def test_year_only_request_rescues_to_a_filter_because_text_match_rejects_it(self):
        p = _plan()
        request = 'songs from 1985'
        plan = p._synthesize_rescue_plan(request, p.extract_hints(request), [])
        assert plan.primaries == []
        assert plan.filter['year_min'] == 1985

    def test_rescue_never_synthesizes_an_unconstrained_search_database(self):
        p = _plan()
        for request in ('music', 'something good', 'surprise me'):
            plan = p._synthesize_rescue_plan(request, p.extract_hints(request), [])
            assert all(c['name'] != 'search_database' for c in plan.primaries)

    def test_unsupported_duration_note_survives_the_rescue(self, monkeypatch):
        p = _plan()
        result, _seen, _logs = _run_plan(
            p, monkeypatch, 'short punchy songs under 3 minutes',
            [{'name': 'search_database', 'arguments': {}}],
            raw_request='short punchy songs under 3 minutes',
        )
        assert result['songs']
        assert any('duration' in n for n in result['plan_notes'])

    def test_at_most_one_extra_llm_call_across_the_empty_plan_path(self, monkeypatch):
        p = _plan()
        seen_prompts = []

        def plans(user_message):
            seen_prompts.append(user_message)
            return {'tool_calls': [{'name': 'search_database', 'arguments': {}}]}

        result, _seen, _logs = _run_plan(p, monkeypatch, 'music', plans, raw_request='music')
        assert 'error' not in result
        assert len(seen_prompts) == 2
        assert 'PREVIOUS ATTEMPT FAILED' in seen_prompts[1]

    def test_provider_error_still_produces_songs_with_a_plan_note(self, monkeypatch):
        p = _plan()

        def plans(_user_message):
            return {'error': 'connection refused'}

        result, seen, _logs = _run_plan(
            p, monkeypatch, 'dark moody electronic', plans,
            raw_request='dark moody electronic',
        )
        assert 'error' not in result
        assert result['songs']
        assert any('unreachable' in n for n in result['plan_notes'])
        assert any(n == 'text_match' for n, _ in seen)


class TestKnowledgeLookupGrounding:
    def test_positive_filter_is_injected_into_the_call_not_discarded(self, monkeypatch):
        p = _plan()
        _result, seen, _logs = _run_plan(
            p, monkeypatch,
            'the most popular hip hop hits of the 2010s, only high energy ones',
            [
                {'name': 'knowledge_lookup',
                 'arguments': {'user_request': 'most popular hip hop hits of the 2010s'}},
                {'name': 'search_database',
                 'arguments': {'genres': ['Hip-Hop'], 'energy_min': 0.7,
                               'year_min': 2010, 'year_max': 2019}},
            ])
        kl = [a for n, a in seen if n == 'knowledge_lookup']
        assert kl and kl[0]['grounding_filter']['genres'] == ['Hip-Hop']
        assert kl[0]['grounding_filter']['year_min'] == 2010

    def test_exclusions_are_forwarded_to_the_brainstorm_as_a_gate(self, monkeypatch):
        p = _plan()
        _result, seen, _logs = _run_plan(
            p, monkeypatch,
            'greatest disco hits of the 70s, but absolutely nothing by the Bee Gees',
            [
                {'name': 'knowledge_lookup',
                 'arguments': {'user_request': 'greatest disco hits of the 70s'}},
                {'name': 'search_database',
                 'arguments': {'exclude_artists': ['Bee Gees']}},
            ])
        kl = [a for n, a in seen if n == 'knowledge_lookup']
        assert kl and kl[0]['gate_filter']['exclude_artists'] == ['Bee Gees']

    def test_plan_filter_cleared_so_the_rerank_never_touches_brainstorm_output(self, monkeypatch):
        p = _plan()
        result, seen, _logs = _run_plan(
            p, monkeypatch, 'famous 90s hits',
            [
                {'name': 'knowledge_lookup', 'arguments': {'user_request': 'famous 90s hits'}},
                {'name': 'search_database', 'arguments': {'year_min': 1990, 'year_max': 1999}},
            ])
        assert result['filter_applied'] is False
        assert all(n != 'search_database' for n, _ in seen)

    def test_hint_backstop_now_runs_for_knowledge_lookup_plans(self, monkeypatch):
        p = _plan()
        _result, seen, _logs = _run_plan(
            p, monkeypatch, 'famous hits from the 90s',
            [{'name': 'knowledge_lookup',
              'arguments': {'user_request': 'famous hits from the 90s'}}])
        kl = [a for n, a in seen if n == 'knowledge_lookup']
        assert kl and kl[0]['grounding_filter']['year_min'] == 1990

    def test_artist_is_dropped_from_the_grounding_with_a_plan_note(self, monkeypatch):
        p = _plan()
        result, seen, _logs = _run_plan(
            p, monkeypatch, 'best of the 90s',
            [
                {'name': 'knowledge_lookup', 'arguments': {'user_request': 'best of the 90s'}},
                {'name': 'search_database', 'arguments': {'artist': 'Oasis'}},
            ])
        kl = [a for n, a in seen if n == 'knowledge_lookup']
        assert kl
        assert 'artist' not in (kl[0].get('grounding_filter') or {})
        assert 'artist' not in (kl[0].get('gate_filter') or {})
        assert any('Oasis' in n for n in result['plan_notes'])


class TestThreeToolExampleIsOffered:
    def test_examples_include_exactly_one_three_tool_plan(self):
        pr = _prompts()
        examples = pr._build_examples(_tools_fixture())
        three = [e for e in examples if len(_example_calls(e)) == 3]
        assert len(three) == 1
        assert [c['name'] for c in _example_calls(three[0])] == [
            'seed_search', 'text_match', 'search_database',
        ]

    def test_three_tool_example_is_dropped_when_text_match_is_absent(self):
        pr = _prompts()
        tools = [t for t in _tools_fixture() if t['name'] != 'text_match']
        examples = pr._build_examples(tools)
        assert all(len(_example_calls(e)) <= 2 for e in examples)

    def test_planning_rule_no_longer_anchors_to_a_single_finder(self):
        pr = _prompts()
        prompt = pr.build_mcp_system_prompt(_tools_fixture())
        assert 'one finder plus one filter is the usual plan' not in prompt
        assert 'one finder tool per distinct part' in prompt


class TestPlannerLogLinesStayFrontendParsable:
    DIMENSION_RE = re.compile(
        r'^(genres|voices|moods|other_features|energy|tempo|year|min_rating|scale|key|'
        r'artist|album)\s+.*->'
    )

    def _all_logs(self, monkeypatch):
        p = _plan()
        logs_all = []
        cases = [
            ('music', [{'name': 'search_database', 'arguments': {}}]),
            ('like Miles Davis, nothing after 1970', [
                {'name': 'seed_search',
                 'arguments': {'seeds': [{'type': 'artist', 'name': 'Miles Davis'}]}},
                {'name': 'search_database',
                 'arguments': {'exclude_artists': ['Miles Davis'],
                               'exclude_genres': ['jazz', 'jazz'], 'year_max': 1970}},
            ]),
            ('famous 90s hits', [
                {'name': 'knowledge_lookup', 'arguments': {'user_request': 'famous 90s hits'}},
                {'name': 'search_database', 'arguments': {'year_min': 1990, 'year_max': 1999}},
            ]),
        ]
        for request, calls in cases:
            _r, _s, logs = _run_plan(p, monkeypatch, request, calls, raw_request=request)
            logs_all.extend(logs)
        return logs_all

    def test_no_new_log_line_is_mistaken_for_a_filter_dimension(self, monkeypatch):
        matched = [
            line for line in self._all_logs(monkeypatch)
            if self.DIMENSION_RE.match(line.strip())
        ]

        assert matched
        for line in matched:
            assert line.startswith('   '), line

    def test_only_real_tool_executions_emit_a_tool_or_primary_prefix(self, monkeypatch):
        prefixed = [
            line.strip() for line in self._all_logs(monkeypatch)
            if line.strip().startswith(('TOOL:', 'PRIMARY:'))
        ]
        assert prefixed
        for line in prefixed:
            assert line.split(':', 1)[1].strip() in {
                'seed_search', 'text_match', 'knowledge_lookup', 'search_database',
            }
