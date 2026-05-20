import pytest

from tasks.smart_session_builder import (
    SMART_SESSION_DEFAULT_LENGTH,
    SMART_SESSION_MAX_LENGTH,
    SMART_SESSION_MIN_LENGTH,
    SmartSessionValidationError,
    build_smart_session_preview,
    normalize_avoid_rules,
    normalize_anchor,
    validate_export_request,
    validate_preview_request,
)


def test_preview_requires_prompt_or_anchor():
    with pytest.raises(SmartSessionValidationError, match='prompt'):
        validate_preview_request({'prompt': '   ', 'anchors': []})


def test_preview_clamps_length_and_max_per_artist():
    data = validate_preview_request({
        'prompt': 'quiet synths',
        'length': 999,
        'max_per_artist': 0,
    })

    assert data['length'] == SMART_SESSION_MAX_LENGTH
    assert data['max_per_artist'] == 1


def test_preview_uses_default_length():
    data = validate_preview_request({'prompt': 'morning acoustic'})

    assert data['length'] == SMART_SESSION_DEFAULT_LENGTH


def test_preview_clamps_short_length():
    data = validate_preview_request({'prompt': 'short set', 'length': 1})

    assert data['length'] == SMART_SESSION_MIN_LENGTH


def test_normalize_anchor_accepts_song_anchor():
    anchor = normalize_anchor({'type': 'song', 'item_id': ' track-1 ', 'weight': 1.5})

    assert anchor == {'type': 'song', 'item_id': 'track-1', 'weight': 1.0}


def test_normalize_anchor_rejects_unknown_type():
    with pytest.raises(SmartSessionValidationError, match='Only song anchors'):
        normalize_anchor({'type': 'artist', 'item_id': 'artist-1'})


def test_normalize_avoid_rules_deduplicates_values():
    rules = normalize_avoid_rules({
        'artists': [' Alice ', 'alice', 'Bob'],
        'terms': 'live',
    })

    assert rules == {'artists': ['Alice', 'Bob'], 'terms': ['live']}


def test_build_preview_returns_day_one_placeholder_shape():
    preview = build_smart_session_preview({'prompt': 'warm dusk songs'})

    assert preview['session_id'] is None
    assert preview['playlist_name'] == 'Smart Session - warm dusk songs'
    assert preview['tracks'] == []
    assert preview['warnings']


def test_export_validation_deduplicates_track_ids():
    payload = validate_export_request({
        'playlist_name': 'My Session',
        'track_ids': ['a', 'a', ' b ', ''],
    })

    assert payload == {'playlist_name': 'My Session', 'track_ids': ['a', 'b']}