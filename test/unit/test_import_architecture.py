# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Module-level import architecture guardrails across the codebase.

Builds an eager-import graph via AST and asserts the layering, acyclicity, and
independence rules that keep import time and coupling under control.

Main Features:
* Foundation modules stay leaves and there are no module-level import cycles
* Eager import chains stay within the depth ceiling
* Layered dependencies point downward, forbidden edges are absent, and the
  independent app_* blueprints do not cross-import
"""

import ast
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv-windows",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    "pginstall",
    "native-build",
    "test",
}

LEAF_MODULES = {
    "config",
    "tz_helper",
    "error.error_dictionary",
    "ssrf_guard",
    "sanitization",
    "tasks.memory_utils",
}

ALLOWED_CYCLES = {
    frozenset({"lyrics", "lyrics.lyrics_transcriber"}),
    frozenset({"error", "error.error_manager"}),
}

MAX_CHAIN = 5

LAYERS = [
    {
        "config",
        "tz_helper",
        "error.error_dictionary",
        "ssrf_guard",
        "sanitization",
        "tasks.memory_utils",
    },
    {
        "database",
        "taskqueue",
        "tasks.ai.prompts",
        "tasks.ai.providers.openai",
        "tasks.ai.providers.gemini",
        "tasks.ai.providers.mistral",
    },
    {"app_helper", "app_helper_artist", "tasks.ai.api"},
    {"tasks.clustering_helper", "tasks.analysis_helper"},
    {"tasks.clustering", "tasks.analysis"},
    {"app"},
]

FORBIDDEN_IMPORTS = [
    ("database", "app_helper"),
    ("taskqueue", "app_helper"),
    ("database", "app_helper_artist"),
    ("taskqueue", "app_helper_artist"),
    ("tasks.ai.prompts", "tasks.ai.api"),
    ("tasks.ai.providers.openai", "tasks.ai.api"),
    ("tasks.ai.providers.gemini", "tasks.ai.api"),
    ("tasks.ai.providers.mistral", "tasks.ai.api"),
    ("app_helper", "tasks.clustering"),
    ("app_helper", "tasks.analysis"),
]

INDEPENDENT_GROUPS = [
    {
        "app_chat",
        "app_clustering",
        "app_analysis",
        "app_cron",
        "app_ivf",
        "app_sonic_fingerprint",
        "app_path",
        "app_external",
        "app_alchemy",
        "app_map",
        "app_artist_similarity",
        "app_clap_search",
        "app_lyrics",
        "app_sem_grove",
        "app_backup",
        "app_provider_migration",
        "app_dashboard",
        "app_users",
        "app_sync",
    },
]


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
                base = (
                    _resolve_relative(node.module or "", node.level, name, is_package)
                    if node.level
                    else (node.module or "")
                )
                targets = (
                    [f"{base}.{alias.name}" for alias in node.names if base]
                    if base
                    else [alias.name for alias in node.names]
                )
            else:
                continue
            for target in targets:
                parts = target.split(".")
                for i in range(len(parts), 0, -1):
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
    color = {}
    dag = defaultdict(set)

    def _strip(u):
        color[u] = 1
        for v in sorted(graph.get(u, ())):
            c = color.get(v, 0)
            if c == 1:
                continue
            dag[u].add(v)
            if c == 0:
                _strip(v)
        color[u] = 2

    for root in sorted(modules):
        if color.get(root, 0) == 0:
            _strip(root)

    cache = {}

    def depth_from(node):
        if node in cache:
            return cache[node]
        best = (1, (node,))
        for succ in dag.get(node, ()):
            sub_len, sub_chain = depth_from(succ)
            if 1 + sub_len > best[0]:
                best = (1 + sub_len, (node,) + sub_chain)
        cache[node] = best
        return best

    overall = (0, ())
    for node in modules:
        candidate = depth_from(node)
        if candidate[0] > overall[0]:
            overall = candidate
    return overall


def _max_chains(graph, modules):
    best_len = 0
    chains = []

    def depth_first(node, path):
        nonlocal best_len, chains
        extended = False
        for succ in sorted(graph.get(node, ())):
            if succ in path:
                continue
            extended = True
            depth_first(succ, path + (succ,))
        if not extended:
            n = len(path)
            if n > best_len:
                best_len, chains = n, [path]
            elif n == best_len:
                chains.append(path)

    for node in sorted(modules):
        depth_first(node, (node,))
    return best_len, sorted(set(chains))


@lru_cache(maxsize=1)
def _graph():
    modules = _collect_modules()
    return modules, _build_eager_graph(modules)


def architecture_report():
    modules, graph = _graph()
    level = {m: i for i, layer in enumerate(LAYERS) for m in layer}

    down = horiz = up = 0
    for src, dsts in graph.items():
        if src not in level:
            continue
        for dst in dsts:
            if dst not in level:
                continue
            if level[dst] > level[src]:
                up += 1
            elif level[dst] == level[src]:
                horiz += 1
            else:
                down += 1

    max_len, chains = _max_chains(graph, modules)

    lines = [
        "Layers (L0 = foundation, ascending to the app entrypoint); "
        "every dependency must point DOWN to a lower or equal layer:"
    ]
    for i, layer in enumerate(LAYERS):
        lines.append(f"  L{i}: " + ", ".join(sorted(layer)))
    lines.append(
        f"  layered edges: {down} downward (ok), {horiz} horizontal/same-layer, "
        f"{up} upward (ILLEGAL)"
    )
    lines.append("")
    status = "OK" if max_len <= MAX_CHAIN else "OVER CEILING"
    lines.append(
        f"Max eager import chain: {max_len} modules (ceiling MAX_CHAIN={MAX_CHAIN}) -> {status}"
    )
    lines.append(f"Chains at depth {max_len} ({len(chains)}):")
    for chain in chains:
        lines.append("  " + " -> ".join(chain))
    return lines


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
    length, _ = _longest_chain(graph, modules)
    if length > MAX_CHAIN:
        max_len, chains = _max_chains(graph, modules)
        recap = "\n  ".join(" -> ".join(c) for c in chains)
        raise AssertionError(
            f"Eager import chain of {length} modules exceeds the maximum of "
            f"{MAX_CHAIN}. Convert one edge in each NEW chain below to a "
            f"function-level import to flatten it.\n"
            f"All chains at depth {max_len}:\n  {recap}"
        )


def _reachable(graph, start):
    seen = set()
    stack = list(graph.get(start, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    return seen


def test_layered_dependencies_point_downward():
    modules, graph = _graph()
    level = {m: i for i, layer in enumerate(LAYERS) for m in layer}
    unknown = sorted(m for m in level if m not in modules)
    assert not unknown, f"LAYERS references modules that no longer exist: {unknown}"

    violations = []
    for src, src_level in level.items():
        for dst in _reachable(graph, src):
            if dst in level and level[dst] > src_level:
                violations.append(f"{src} (layer {src_level}) -> {dst} (layer {level[dst]})")
    assert not violations, (
        "Lower layers must not import higher layers at module level (move the "
        "dependency down, or push the import inside the function that uses it):\n  "
        + "\n  ".join(sorted(violations))
    )


def test_forbidden_imports():
    modules, graph = _graph()
    violations = []
    for src, dst in FORBIDDEN_IMPORTS:
        if src in modules and dst in _reachable(graph, src):
            violations.append(f"{src} -> ... -> {dst}")
    assert not violations, (
        "Forbidden module-level dependencies detected (these layers must not "
        "depend on the higher ones):\n  " + "\n  ".join(violations)
    )


def test_independent_modules_do_not_cross_import():
    modules, graph = _graph()
    violations = []
    for group in INDEPENDENT_GROUPS:
        present = group & set(modules)
        for src in sorted(present):
            for dst in sorted(_reachable(graph, src) & present - {src}):
                violations.append(f"{src} -> {dst}")
    assert not violations, (
        "Independent modules must not import one another at module level "
        "(compose them through app.py instead):\n  " + "\n  ".join(violations)
    )
