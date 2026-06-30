import importlib
import sys
import types


def _ensure_config_stub():
    if 'config' in sys.modules:
        return
    cfg = types.ModuleType('config')
    cfg.MOOD_LABELS = [
        'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists',
        'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal',
        'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock',
        'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
        'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party',
        'country', 'easy listening', 'sexy', 'catchy', 'funk', 'electro',
        'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad',
        'House', 'happy',
    ]
    cfg.OTHER_FEATURE_LABELS = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']
    cfg.STRATIFIED_GENRES = [
        'rock', 'pop', 'alternative', 'indie', 'electronic', 'jazz', 'metal',
        'classic rock', 'soul', 'indie rock', 'electronica', 'folk', 'punk',
        'blues', 'hard rock', 'ambient', 'acoustic', 'experimental', 'Hip-Hop',
        'country', 'funk', 'electro', 'heavy metal', 'Progressive rock', 'rnb',
        'indie pop', 'House',
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
        assert result['mood_vector'] == ['happy']

    def test_energy_phrase_extracted(self):
        v = _vocab()
        result = v.normalize_mood_list(['chill'])
        assert result['energy_min'] == 0.0
        assert result['energy_max'] == 0.35

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
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'search_database', 'arguments': {'moods': ['female voice', 'happy']}}
        ])
        assert plan_obj.filter is not None
        assert 'female vocalists' in plan_obj.filter.get('voices', [])
        assert 'happy' in plan_obj.filter.get('moods', [])

    def test_voices_field_passes_through(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'search_database', 'arguments': {'voices': ['female voice']}}
        ])
        assert 'female vocalists' in plan_obj.filter['voices']

    def test_short_female_token_remapped(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'search_database', 'arguments': {'moods': ['female']}}
        ])
        assert 'female vocalists' in plan_obj.filter.get('voices', [])
        assert 'female vocalist' in plan_obj.filter.get('voices', [])

    def test_non_canonical_genre_dropped_with_note(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'search_database', 'arguments': {'genres': ['rock', 'definitelynotagenre']}}
        ])
        assert 'rock' in plan_obj.filter['genres']
        assert any("dropped" in n.lower() or "unrecognized" in n.lower() for n in plan_obj.notes)


class TestValidatePlanArgs:
    def test_drops_seed_search_with_no_usable_seeds(self):
        p = _plan()
        log = []
        out = p.validate_plan_args(
            [{'name': 'seed_search', 'arguments': {'seeds': [{'type': 'song', 'title': '', 'song_artist': 'X'}]}}],
            user_wants_rating=False, log_messages=log,
        )
        assert out == []

    def test_coerces_single_seed_alchemy_to_union(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'seed_search', 'arguments': {
                'seeds': [{'type': 'artist', 'name': 'Madonna'}],
                'blend_mode': 'alchemy',
            }}],
            user_wants_rating=False,
        )
        assert out[0]['name'] == 'seed_search'
        assert out[0]['arguments']['blend_mode'] == 'union'
        assert out[0]['arguments']['seeds'][0]['name'] == 'Madonna'

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
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'blink-182'}]}},
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'artist', 'name': 'Green Day'}]}},
        ])
        assert len(plan_obj.primaries) == 2
        assert plan_obj.filter is None

    def test_primaries_plus_filter(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'seed_search', 'arguments': {'seeds': [{'type': 'song', 'title': 'By The Way', 'artist': 'Red Hot Chili Peppers'}]}},
            {'name': 'search_database', 'arguments': {'voices': ['female voice']}},
        ])
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

    def test_energy_phrase_detected(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints
        h = extract_hints("chill jazz")
        assert h['energy_min'] == 0.0
        assert h['energy_max'] == 0.35

    def test_explicit_energy_floor(self):
        _ensure_config_stub()
        from tasks.ai.planner import extract_hints
        h = extract_hints("songs with energy above 0.5")
        assert h['energy_min'] == 0.5


class TestMultiPrimaryClassifier:
    def _tools(self):
        return [
            {'name': 'seed_search'},
            {'name': 'text_match'},
            {'name': 'knowledge_lookup'},
            {'name': 'search_database'},
        ]

    @staticmethod
    def _names(tools):
        return {t['name'] for t in tools}

    def test_new_shape_passthrough(self):
        p = _plan()
        out = p._normalize_classifier_result({'primaries': ['text'], 'needs_filter': True})
        assert out == {'primaries': ['text'], 'needs_filter': True}

    def test_dedupe_drop_invalid_preserve_order(self):
        p = _plan()
        out = p._normalize_classifier_result(
            {'primaries': ['text', 'text', 'bogus', 'seed'], 'needs_filter': False}
        )
        assert out == {'primaries': ['text', 'seed'], 'needs_filter': False}

    def test_pure_metadata_passthrough(self):
        p = _plan()
        out = p._normalize_classifier_result({'primaries': [], 'needs_filter': True})
        assert out == {'primaries': [], 'needs_filter': True}

    def test_degenerate_empty_returns_none(self):
        p = _plan()
        assert p._normalize_classifier_result({'primaries': [], 'needs_filter': False}) is None

    def test_needs_filter_string_coerced(self):
        p = _plan()
        out = p._normalize_classifier_result({'primaries': ['seed'], 'needs_filter': 'true'})
        assert out['needs_filter'] is True

    def test_lone_string_primaries_coerced(self):
        p = _plan()
        out = p._normalize_classifier_result({'primaries': 'seed', 'needs_filter': False})
        assert out == {'primaries': ['seed'], 'needs_filter': False}

    def test_legacy_metadata_translated(self):
        p = _plan()
        out = p._normalize_classifier_result({'intent': 'metadata', 'needs_filter': False})
        assert out == {'primaries': [], 'needs_filter': True}

    def test_legacy_seed_translated(self):
        p = _plan()
        out = p._normalize_classifier_result({'intent': 'seed', 'needs_filter': True})
        assert out == {'primaries': ['seed'], 'needs_filter': True}

    def test_non_dict_returns_none(self):
        p = _plan()
        assert p._normalize_classifier_result(None) is None
        assert p._normalize_classifier_result("text") is None

    def test_tools_text_plus_filter(self):
        p = _plan()
        out = p.tools_for_intent(['text'], True, self._tools())
        assert self._names(out) == {'text_match', 'search_database'}

    def test_tools_two_primaries_no_filter(self):
        p = _plan()
        out = p.tools_for_intent(['seed', 'text'], False, self._tools())
        assert self._names(out) == {'seed_search', 'text_match'}

    def test_tools_empty_primaries_with_filter(self):
        p = _plan()
        out = p.tools_for_intent([], True, self._tools())
        assert self._names(out) == {'search_database'}

    def test_tools_knowledge_plus_filter(self):
        p = _plan()
        out = p.tools_for_intent(['knowledge'], True, self._tools())
        assert self._names(out) == {'knowledge_lookup', 'search_database'}

    def test_tools_unresolvable_falls_back_to_full(self):
        p = _plan()
        tools = [{'name': 'text_match'}, {'name': 'seed_search'}]
        out = p.tools_for_intent(['knowledge'], False, tools)
        assert out == tools
