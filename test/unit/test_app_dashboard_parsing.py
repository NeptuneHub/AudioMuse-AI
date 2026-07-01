# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Key-value parsing for the dashboard's mood and feature strings.

Covers _parse_keyval, which turns "label:score" comma lists into a float map
while tolerating malformed input.

Main Features:
* Empty and None inputs yield an empty dict
* Pairs with a non-numeric value, no separator, or empty key are skipped
* Whitespace around keys and values is stripped before parsing
"""

from app_dashboard import _parse_keyval


class TestParseKeyval:
    def test_empty_string_returns_empty_dict(self):
        assert _parse_keyval('') == {}

    def test_none_returns_empty_dict(self):
        assert _parse_keyval(None) == {}

    def test_invalid_value_pair_is_skipped(self):
        result = _parse_keyval('key1:invalid,key2:5.5')
        assert 'key1' not in result
        assert result == {'key2': 5.5}

    def test_pair_without_separator_is_skipped(self):
        result = _parse_keyval('key1,key2:5.5')
        assert 'key1' not in result
        assert result == {'key2': 5.5}

    def test_valid_multi_pair_string_parses_fully(self):
        result = _parse_keyval('rock:0.9,jazz:0.05,pop:0.12')
        assert result == {'rock': 0.9, 'jazz': 0.05, 'pop': 0.12}

    def test_key_whitespace_is_stripped(self):
        result = _parse_keyval(' rock :0.9, jazz:0.1')
        assert result == {'rock': 0.9, 'jazz': 0.1}

    def test_empty_key_is_skipped(self):
        result = _parse_keyval(':0.5,pop:0.2')
        assert result == {'pop': 0.2}

    def test_value_with_surrounding_whitespace_parses(self):
        result = _parse_keyval('rock: 0.9 ')
        assert result == {'rock': 0.9}
