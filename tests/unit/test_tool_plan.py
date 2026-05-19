"""Unit tests for the single-pass plan validation + vocab normalization layer."""
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
        assert result['mood_vector'] == []

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
        assert result['voices'] == ['female vocalists', 'male vocalists']
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
    def test_drops_empty_song_similarity(self):
        p = _plan()
        log = []
        out = p.validate_plan_args(
            [{'name': 'song_similarity', 'arguments': {'song_title': '', 'song_artist': 'X'}}],
            user_wants_rating=False, log_messages=log,
        )
        assert out == []

    def test_coerces_single_alchemy_to_artist_similarity(self):
        p = _plan()
        out = p.validate_plan_args(
            [{'name': 'song_alchemy', 'arguments': {'add_items': [{'type': 'artist', 'id': 'Madonna'}]}}],
            user_wants_rating=False,
        )
        assert out[0]['name'] == 'artist_similarity'
        assert out[0]['arguments']['artist'] == 'Madonna'

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
            {'name': 'artist_similarity', 'arguments': {'artist': 'blink-182'}},
            {'name': 'artist_similarity', 'arguments': {'artist': 'Green Day'}},
        ])
        assert len(plan_obj.primaries) == 2
        assert plan_obj.filter is None

    def test_primaries_plus_filter(self):
        p = _plan()
        plan_obj = p.validate_and_normalize_plan([
            {'name': 'song_similarity', 'arguments': {'song_title': 'By The Way', 'song_artist': 'Red Hot Chili Peppers'}},
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
