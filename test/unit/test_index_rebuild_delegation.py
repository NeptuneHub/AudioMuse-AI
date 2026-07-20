# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Static check that cleaning runs the shared final rebuild INLINE, like analysis.

Cleaning changes what each server maps (unbind) and can remove catalogue rows
(orphan delete), so it must refresh the similarity indexes before it reports
complete. It does this the same way analysis finalizes: by calling the shared
_run_all_index_builds inline (which also publishes the 'reload' Flask listens
for), NOT by re-implementing the eight build_and_store_* builders itself. Parses
tasks/cleaning.py with AST to confirm both: it calls _run_all_index_builds, and
it never calls a partial builder directly.

Main Features:
* The cleaning task calls _run_all_index_builds inline (shared with analysis)
* It calls no build_and_store_* builder directly: cleaning owns no rebuild logic
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


class TestCleaningRebuildsIndexesInlineLikeAnalysis:
    def test_cleaning_calls_the_shared_rebuild_inline(self):
        funcs = _function_defs("tasks/cleaning.py")
        assert "identify_and_clean_orphaned_albums_task" in funcs
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        assert "_run_all_index_builds" in called, (
            "cleaning must run the shared final rebuild inline so it completes only "
            "once every server's similarity results reflect the cleaned catalogue"
        )

    def test_cleaning_never_calls_a_partial_builder_inline(self):
        funcs = _function_defs("tasks/cleaning.py")
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        leaked = sorted(b for b in _PARTIAL_BUILDERS if b in called)
        assert not leaked, (
            f"these builders are called directly: {leaked}. Cleaning must reuse "
            "_run_all_index_builds, not re-implement the builders in its own job."
        )
