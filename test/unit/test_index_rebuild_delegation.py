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


class TestCleaningDelegatesRebuild:
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
