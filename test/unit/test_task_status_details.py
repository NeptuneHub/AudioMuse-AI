# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Coercion of task-status details from stored JSON into a dict.

Covers app_helper.coerce_db_details normalizing the details column, which may
arrive as a dict, a JSON string or an invalid value.

Main Features:
* An existing dict passes through by identity without re-parsing
* A JSON string is parsed to a dict
* None, empty, invalid JSON and non-string/non-dict inputs yield an empty dict
"""

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
