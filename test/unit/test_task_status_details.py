"""Unit coverage for app_helper.coerce_db_details.

This is the exact detail-normalization the /api/status/<task_id> endpoint relies
on: a TEXT details column comes back as a JSON string (parse it) while a JSONB
column comes back as an already-decoded dict (must NOT be re-parsed). The
integration test proves the real DB hands back those two shapes; this proves the
shared helper maps each shape correctly.
"""
import app_helper


def test_dict_passthrough_is_identity_no_reparse():
    # JSONB path: psycopg2 already decoded it; the helper returns it untouched.
    d = {"log": ["Analyzing album", "Done"], "nested": {"a": 1, "b": [2, 3]}}
    assert app_helper.coerce_db_details(d) is d


def test_json_string_is_parsed_to_dict():
    # TEXT path: details came back as a JSON string and must be parsed once.
    out = app_helper.coerce_db_details('{"a": 1, "b": [2, 3]}')
    assert out == {"a": 1, "b": [2, 3]}


def test_none_yields_empty_dict():
    assert app_helper.coerce_db_details(None) == {}


def test_empty_string_yields_empty_dict():
    assert app_helper.coerce_db_details('') == {}


def test_invalid_json_yields_empty_dict():
    assert app_helper.coerce_db_details('{not valid json') == {}


def test_non_string_non_dict_yields_empty_dict():
    # A stray int/list is neither a dict nor json.loads-able text -> {}.
    assert app_helper.coerce_db_details(12345) == {}
