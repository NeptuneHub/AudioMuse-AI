# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Regression tests for playlist-curator compatibility after merging main.

Main Features:
* Require media-server imports from the current package layout.
* Guard IVF compatibility and SQL pagination behavior.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_playlist_curator_uses_renamed_mediaserver_package_imports():
    source = (REPO_ROOT / "app_playlist_curator.py").read_text(encoding="utf-8")

    legacy_imports = [
        "tasks.mediaserver_jellyfin",
        "tasks.mediaserver_emby",
        "tasks.mediaserver_navidrome",
        "tasks.mediaserver_lyrion",
    ]

    for legacy_import in legacy_imports:
        assert legacy_import not in source

    assert "tasks.mediaserver.jellyfin" in source
    assert "tasks.mediaserver.emby" in source
    assert "tasks.mediaserver.navidrome" in source
    assert "tasks.mediaserver.lyrion" in source


def test_ivf_manager_no_longer_references_removed_voyager_index_global():
    source = (REPO_ROOT / "tasks" / "ivf_manager.py").read_text(encoding="utf-8")

    assert "voyager_index" not in source


def test_smart_search_filters_are_paged_in_sql():
    source = (REPO_ROOT / "app_playlist_curator.py").read_text(encoding="utf-8")

    assert "SELECT COUNT(*) AS total FROM score WHERE {where_clause}" in source
    assert "SELECT item_id FROM score WHERE {where_clause} ORDER BY item_id LIMIT %s OFFSET %s" in source
    assert "if search_only:\n                offset = (page - 1) * per_page" in source
