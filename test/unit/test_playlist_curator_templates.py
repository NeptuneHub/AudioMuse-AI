# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for playlist-curator template structure and accessibility.

Main Features:
* Require accessible names for curator form controls.
* Keep sidebar navigation ownership in the sidebar partial.
"""

from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH_TEMPLATE = REPO_ROOT / "templates" / "includes" / "_curator_workbench.html"
DEDUP_TEMPLATE = REPO_ROOT / "templates" / "includes" / "_curator_dedup.html"
CURATOR_SHARED_JS = REPO_ROOT / "static" / "playlist_curator" / "curator-shared.js"
CURATOR_EXTENDER_JS = REPO_ROOT / "static" / "playlist_curator" / "curator-extender.js"
CURATOR_TEMPLATE_PATHS = [
    REPO_ROOT / "templates" / "playlist_curator_search.html",
    REPO_ROOT / "templates" / "playlist_curator_extender.html",
    REPO_ROOT / "templates" / "includes" / "_curator_workbench.html",
]


class InputLabelParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.controls = []
        self.label_targets = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"input", "select", "textarea"}:
            self.controls.append((tag, attrs))
        elif tag == "label" and attrs.get("for"):
            self.label_targets.add(attrs["for"])


def test_curator_inputs_have_explicit_labels():
    unlabeled = []
    for path in CURATOR_TEMPLATE_PATHS:
        parser = InputLabelParser()
        parser.feed(path.read_text(encoding="utf-8"))
        for tag, attrs in parser.controls:
            input_type = attrs.get("type", "text")
            input_id = attrs.get("id")
            if input_type == "hidden":
                continue
            has_accessible_name = (
                bool(attrs.get("aria-label"))
                or bool(attrs.get("aria-labelledby"))
                or (input_id in parser.label_targets)
            )
            if not has_accessible_name:
                unlabeled.append(f"{path.relative_to(REPO_ROOT)} {tag}#{input_id or '<missing-id>'}")

    assert unlabeled == []


def test_sidebar_nav_partial_owns_its_list_container():
    sidebar = (REPO_ROOT / "templates" / "sidebar_navi.html").read_text(encoding="utf-8")
    layout = (REPO_ROOT / "templates" / "includes" / "layout.html").read_text(encoding="utf-8")

    assert '<ul class="sidebar-nav">' in sidebar
    assert '<ul class="sidebar-nav">' not in layout


def test_workbench_keeps_create_new_and_adds_contextual_replace_controls():
    template = WORKBENCH_TEMPLATE.read_text(encoding="utf-8")

    assert 'id="curator-wb-name"' in template
    assert 'id="curator-wb-save-btn"' in template
    assert 'id="curator-sheet-name"' in template
    assert 'id="curator-sheet-save-btn"' in template
    assert 'id="curator-wb-replace-btn"' in template
    assert 'id="curator-sheet-replace-btn"' in template
    assert template.count('Replace seeded playlist') == 2


def test_shared_workbench_owns_nonpersistent_seed_target_and_replace_payload():
    source = CURATOR_SHARED_JS.read_text(encoding="utf-8")

    assert "let seededServerPlaylist = null;" in source
    assert "window.curatorSetSeededPlaylistTarget = setSeededPlaylistTarget;" in source
    assert "replace_playlist_name: seededServerPlaylist.playlistName" in source
    assert "unresolvedTracks" in source
    assert "replaceBtn.textContent = `Replace \u201c${seededServerPlaylist.playlistName}\u201d`;" in source
    assert "confirm(message)" in source
    assert "window.curatorReplaceSeededPlaylist = replaceSeededPlaylist;" in source
    assert "localStorage.setItem(STORAGE_KEY, JSON.stringify(workbench))" in source
    assert "JSON.stringify(seededServerPlaylist)" not in source


def test_extender_retains_server_seed_name_and_unresolved_count_for_replacement():
    source = CURATOR_EXTENDER_JS.read_text(encoding="utf-8")

    assert "opt.dataset.playlistName = pl.playlist_name;" in source
    assert "serverSeed = {" in source
    assert "playlistId," in source
    assert "playlistName," in source
    assert "unresolvedTracks: data.unresolved_tracks || 0" in source
    assert "window.curatorSetSeededPlaylistTarget(serverSeed);" in source
    assert source.index("window.curatorSetSeededPlaylistTarget(serverSeed);") < source.index(
        "select.value = SEED_WORKBENCH;"
    )


def test_extender_clears_replacement_target_for_non_server_seed():
    source = CURATOR_EXTENDER_JS.read_text(encoding="utf-8")

    assert "window.curatorSetSeededPlaylistTarget(null);" in source
    assert "seedValue.startsWith('__server__')" in source

def test_duplicate_results_are_announced_as_live_status():
    template = DEDUP_TEMPLATE.read_text(encoding="utf-8")

    assert 'id="curator-dedup-groups" aria-live="polite"' in template
