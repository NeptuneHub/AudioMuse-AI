"""Architecture gate for the module-level import graph.

Only module-level (eager) imports count here; function-level imports are the
sanctioned escape hatch used across the codebase (mediaserver providers,
voyager/app_helper consumers, config's DB-override loader). Three invariants
keep the graph flat and acyclic so deep chains and cycles cannot creep back in:

1. Foundation modules stay leaves: they import nothing internal at module level.
2. No module-level import cycles, except the lyrics package init whose
   try/except fallback design is deliberately order-dependent.
3. No eager import chain may exceed MAX_CHAIN modules (app -> blueprint ->
   hub/manager -> leaf). Anything deeper must use a function-level import.
"""

import ast
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_DIRS = {
    ".git", ".venv", ".venv-windows", "node_modules", "__pycache__",
    "build", "dist", "pginstall", "native-build", "test",
}

LEAF_MODULES = {
    "config",
    "tz_helper",
    "error.error_dictionary",
}

ALLOWED_CYCLES = {
    frozenset({"lyrics", "lyrics.lyrics_transcriber"}),
    frozenset({"error", "error.error_manager"}),
}

MAX_CHAIN = 6  # allows for package __init__.py → submodule edges adding 1-2 phantom hops


def _collect_modules():
    modules = {}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            parts = list(path.relative_to(REPO_ROOT).parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            if not parts:
                continue
            modules[".".join(parts)] = path
    return modules


def _resolve_relative(module, level, current, is_package):
    base = current.split(".") if is_package else current.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _build_eager_graph(modules):
    graph = defaultdict(set)
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        nested = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if child is not node:
                        nested.add(id(child))
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if id(node) in nested:
                continue
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_relative(node.module or "", node.level, name, is_package) if node.level else (node.module or "")
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names if base]
            else:
                continue
            for target in targets:
                parts = target.split(".")
                for i in range(1, len(parts) + 1):
                    candidate = ".".join(parts[:i])
                    if candidate in modules and candidate != name:
                        graph[name].add(candidate)
    return graph


def _find_cycles(graph, modules):
    index = {}
    low = {}
    on_stack = {}
    stack = []
    sccs = []
    counter = [0]
    for root in sorted(modules):
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            node, pointer = work[-1]
            if pointer == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            advanced = False
            successors = sorted(graph.get(node, ()))
            for i in range(pointer, len(successors)):
                succ = successors[i]
                if succ not in index:
                    work[-1] = (node, i + 1)
                    work.append((succ, 0))
                    advanced = True
                    break
                if on_stack.get(succ):
                    low[node] = min(low[node], index[succ])
            if advanced:
                continue
            work.pop()
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    sccs.append(frozenset(component))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return sccs


def _longest_chain(graph, modules):
    cache = {}

    def depth_from(node, path):
        if node in cache:
            return cache[node]
        best = (1, (node,))
        for succ in graph.get(node, ()):
            if succ in path:
                continue
            sub_len, sub_chain = depth_from(succ, path | {node})
            if 1 + sub_len > best[0]:
                best = (1 + sub_len, (node,) + sub_chain)
        if not any(s in path for s in graph.get(node, ())):
            cache[node] = best
        return best

    overall = (0, ())
    for node in modules:
        candidate = depth_from(node, frozenset())
        if candidate[0] > overall[0]:
            overall = candidate
    return overall


@lru_cache(maxsize=1)
def _graph():
    modules = _collect_modules()
    return modules, _build_eager_graph(modules)


def test_foundation_modules_are_leaves():
    _, graph = _graph()
    violations = {leaf: sorted(graph.get(leaf, ())) for leaf in LEAF_MODULES if graph.get(leaf)}
    assert not violations, (
        f"Foundation modules must not import project modules at module level "
        f"(move the import inside the function that uses it): {violations}"
    )


def test_no_module_level_import_cycles():
    modules, graph = _graph()
    cycles = [set(c) for c in _find_cycles(graph, modules) if c not in ALLOWED_CYCLES]
    assert not cycles, (
        f"Module-level import cycles detected (break them with a function-level "
        f"import on one side): {cycles}"
    )


def test_eager_import_chains_stay_shallow():
    modules, graph = _graph()
    length, chain = _longest_chain(graph, modules)
    assert length <= MAX_CHAIN, (
        f"Eager import chain of {length} modules exceeds the maximum of "
        f"{MAX_CHAIN}: {' -> '.join(chain)}. Convert one edge to a "
        f"function-level import to flatten it."
    )
