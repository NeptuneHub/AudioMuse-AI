import app_helper


def test_dict_passthrough_is_identity_no_reparse():
    d = {"log": ["Analyzing album", "Done"], "nested": {"a": 1, "b": [2, 3]}}
    assert app_helper.coerce_db_details(d) is d


def test_json_string_is_parsed_to_dict():
    out = app_helper.coerce_db_details('{"a": 1, "b": [2, 3]}')
    assert out == {"a": 1, "b": [2, 3]}


def test_none_yields_empty_dict():
    assert app_helper.coerce_db_details(None) == {}


def test_empty_string_yields_empty_dict():
    assert app_helper.coerce_db_details('') == {}


def test_invalid_json_yields_empty_dict():
    assert app_helper.coerce_db_details('{not valid json') == {}


def test_non_string_non_dict_yields_empty_dict():
    assert app_helper.coerce_db_details(12345) == {}
