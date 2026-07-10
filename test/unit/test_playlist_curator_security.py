# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Security regression tests for Playlist Curator SQL filter assembly.

Main Features:
* Keep hostile request values in psycopg2 parameters rather than SQL text.
* Reject non-allowlisted fields and operators from structural query fragments.
* Require narrowly scoped Bandit B608 suppressions at trusted assembly sites.
"""

from pathlib import Path

from app_playlist_curator import _build_filter_query


REPO_ROOT = Path(__file__).resolve().parents[2]
CURATOR_SOURCE = REPO_ROOT / "app_playlist_curator.py"


def test_hostile_filter_value_stays_in_bound_parameters():
    hostile = "x%' OR 1=1; DROP TABLE score; --"

    clause, params = _build_filter_query(
        [{"field": "artist", "operator": "contains", "value": hostile}],
        "all",
    )

    assert clause == "(author ILIKE %s)"
    assert hostile not in clause
    assert params == [f"%{hostile}%"]


def test_untrusted_structural_inputs_cannot_enter_filter_clause():
    hostile = "title); DROP TABLE score; --"

    clause, params = _build_filter_query(
        [{"field": hostile, "operator": hostile, "value": "ignored"}],
        hostile,
    )

    assert clause == "1=1"
    assert params == []
    assert hostile not in clause


def test_match_mode_is_reduced_to_fixed_and_or_tokens():
    hostile_mode = "all); DROP TABLE score; --"

    clause, params = _build_filter_query(
        [
            {"field": "artist", "operator": "is", "value": "A"},
            {"field": "album", "operator": "is", "value": "B"},
        ],
        hostile_mode,
    )

    assert clause == "(author = %s OR album = %s)"
    assert hostile_mode not in clause
    assert params == ["A", "B"]


def test_bandit_suppressions_are_limited_to_three_reviewed_queries():
    source = CURATOR_SOURCE.read_text(encoding="utf-8")

    assert source.count("# nosec B608") == 3
    assert "only allowlisted SQL identifiers and operators" in source
