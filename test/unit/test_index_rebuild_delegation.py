# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Static check that cleaning never rebuilds indexes or deletes catalogue rows.

Parses tasks/cleaning.py with AST to confirm the cleanup task only unbinds
server mappings: it must not rebuild any index (the catalogue never changes,
so a rebuild is wasted work) and must not call any partial builder.

Main Features:
* The cleaning task calls neither _run_all_index_builds nor any
  build_and_store_* builder: unbinding mappings never invalidates an index
"""

import ast
import os


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

_PARTIAL_BUILDERS = (
    "build_and_store_ivf_index",
    "build_and_store_clap_index",
    "build_and_store_lyrics_index",
    "build_and_store_lyrics_axes_index",
    "build_and_store_sem_grove_index",
    "build_and_store_artist_index",
    "build_and_store_map_projection",
    "build_and_store_artist_projection",
)


def _function_defs(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(func_node):
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


class TestCleaningNeverRebuildsOrDeletes:
    def test_unbind_only_cleanup_never_rebuilds_indexes(self):
        funcs = _function_defs("tasks/cleaning.py")
        assert "identify_and_clean_orphaned_albums_task" in funcs
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        assert "_run_all_index_builds" not in called

    def test_does_not_call_any_partial_builder(self):
        funcs = _function_defs("tasks/cleaning.py")
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        leaked = sorted(b for b in _PARTIAL_BUILDERS if b in called)
        assert not leaked, (
            f"these builders are called directly: {leaked}. Cleanup unbinds "
            "mappings only; it must never rebuild indexes."
        )
