"""Source-level guard that the post-task index rebuild delegates to the single
shared orchestrator (``tasks.analysis._run_all_index_builds``) rather than the
old partial rebuild path.

History this pins:

* ``collection_manager.sync_collections_task`` used to rebuild ONLY the audio
  IVF index after a sync, leaving CLAP / lyrics / lyrics-axes / SemGrove /
  artist and the 2D map projections stale until the next analysis run.
* ``cleaning.identify_and_clean_orphaned_albums_task`` rebuilt a hand-maintained
  subset of builders inline, in three separate branches.

Both now route through ``_run_all_index_builds``, which builds the full set in
the same order as the analysis task and publishes the reload message. The
runtime behaviour of the orchestrator itself is covered by
``test_index_rebuild_integration.py``; this file only locks in the delegation
contract so a future edit cannot silently revert either task to a partial
rebuild.

These tests parse the source with ``ast`` -- no import of the task modules, no
DB, no ivf/sklearn/librosa -- so they run in every environment and never
skip.
"""

import ast
import os


REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

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
    """Return ``{function_name: FunctionDef}`` for every def in a source file."""
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(func_node):
    """Return the set of callee names (bare and attribute) inside a function."""
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


class TestCleaningDelegatesRebuild:
    """tasks/cleaning.py :: identify_and_clean_orphaned_albums_task"""

    def test_calls_run_all_index_builds(self):
        funcs = _function_defs("tasks/cleaning.py")
        assert "identify_and_clean_orphaned_albums_task" in funcs
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        assert "_run_all_index_builds" in called

    def test_does_not_hand_maintain_a_builder_subset(self):
        funcs = _function_defs("tasks/cleaning.py")
        called = _called_names(funcs["identify_and_clean_orphaned_albums_task"])
        leaked = sorted(b for b in _PARTIAL_BUILDERS if b in called)
        assert not leaked, (
            f"these builders are called directly: {leaked}. The rebuild must go "
            "through _run_all_index_builds so the full set always stays in sync."
        )
