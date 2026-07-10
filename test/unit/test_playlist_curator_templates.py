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
