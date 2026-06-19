"""Architecture gate for the module-level import graph.

Only module-level (eager) imports count here; function-level imports are the
sanctioned escape hatch used across the codebase (mediaserver providers,
ivf/app_helper consumers, config's DB-override loader). Six invariants
keep the graph flat, layered, and acyclic so deep chains and cycles cannot
creep back in:

1. Foundation modules stay leaves: they import nothing internal at module level.
2. No module-level import cycles, except the lyrics package init whose
   try/except fallback design is deliberately order-dependent.
3. No eager import chain may exceed MAX_CHAIN modules (e.g. app -> blueprint ->
   helper -> service-facade -> leaf). Anything deeper must use a function-level
   import.
4. Layers point downward: a module may import its own or any lower layer, never
   a higher one (checked transitively, so indirect chains count too).
5. Forbidden edges: specific module-level dependencies are banned outright
   (e.g. the database/queue layer must never import the app_helper facade).
6. Independence: route blueprints never import one another -- they compose only
   through app.py.
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
    "ssrf_guard",
    "sanitization",
    "tasks.memory_utils",
}

ALLOWED_CYCLES = {
    frozenset({"lyrics", "lyrics.lyrics_transcriber"}),
    frozenset({"error", "error.error_manager"}),
}

# The honest eager-import floor for this codebase is 5, not 4: two intentional
# 3-deep facades are each imported one hop below the root task/blueprint modules.
#   - error/ package: error -> error.error_manager -> error.error_dictionary
#   - AI dispatch:     tasks.ai.api -> {providers.openai, prompts} -> config
# Flattening either to reach 4 would dismantle a deliberate facade (merging the
# error dictionary/manager, or inlining the AI providers) and hurt readability,
# so 5 is the enforced ceiling. Anything deeper must use a function-level import.
MAX_CHAIN = 5

# --- Layered architecture (invariant 4) ------------------------------------
# Ordered low -> high. A module may import its own layer or any LOWER layer,
# never a higher one (checked transitively). Only listed modules are
# constrained; cycle-participating modules (error/, lyrics/) are intentionally
# omitted and governed by the cycles test instead.
LAYERS = [
    {"config", "tz_helper", "error.error_dictionary", "ssrf_guard", "sanitization", "tasks.memory_utils"},
    {"database", "taskqueue", "tasks.ai.prompts",
     "tasks.ai.providers.openai", "tasks.ai.providers.gemini", "tasks.ai.providers.mistral"},
    {"app_helper", "app_helper_artist", "tasks.ai.api"},
    {"tasks.clustering_helper", "tasks.analysis_helper"},
    {"tasks.clustering", "tasks.analysis"},
    {"app"},
]

# --- Forbidden module-level dependencies (invariant 5) ---------------------
# (importer, target): the importer must not reach the target via any eager
# import chain (direct or indirect). Keeps the data/queue layer and the AI
# transport modules from depending on the higher facade/domain layers.
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

# --- Independence groups (invariant 6) -------------------------------------
# Members must not import one another (direct or indirect). Flask route
# blueprints are self-contained features that compose only through app.py.
INDEPENDENT_GROUPS = [
    {"app_chat", "app_clustering", "app_analysis", "app_cron", "app_ivf",
     "app_sonic_fingerprint", "app_path", "app_external", "app_alchemy", "app_map",
     "app_waveform", "app_artist_similarity", "app_clap_search", "app_lyrics",
     "app_sem_grove", "app_backup", "app_provider_migration", "app_dashboard",
     "app_users", "app_sync"},
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
                base = _resolve_relative(node.module or "", node.level, name, is_package) if node.level else (node.module or "")
                # Track the imported names (base.name). The ancestor-package
                # loop below also charges the base package itself, because
                # importing base.name runs base/__init__.py first.
                targets = [f"{base}.{alias.name}" for alias in node.names if base] if base else [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                parts = target.split(".")
                # Importing "a.b.c" (or "from a.b import c") runs a/__init__.py
                # and a/b/__init__.py before binding the target, so the importer
                # eagerly depends on every ancestor package that is a real module
                # AND the deepest matching module. Empty package __init__ files
                # are graph leaves and add no depth; only packages that import at
                # module level (e.g. error/ and lyrics/) actually extend a chain.
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
    """Longest eager-import chain (module count), with each module counted once.

    The graph may contain the allowed package-init cycles (error/, lyrics/), so a
    naive longest-path walk could revisit a node and over-count. We first strip
    cycle back-edges with a DFS to obtain a DAG, then take its longest path with
    memoization. Stripping back-edges keeps every chain a simple path while
    preserving the honest forward depth, e.g.
    app -> app_setup -> error -> error.error_manager -> error.error_dictionary.
    """
    # Strip back-edges (edges into the current DFS stack) so cycles cannot
    # inflate a chain by revisiting a module.
    color = {}  # 0 = unvisited, 1 = on stack, 2 = done
    dag = defaultdict(set)

    def _strip(u):
        color[u] = 1
        for v in sorted(graph.get(u, ())):
            c = color.get(v, 0)
            if c == 1:
                continue  # back-edge into the DFS stack (a cycle): drop it
            dag[u].add(v)
            if c == 0:
                _strip(v)
        color[u] = 2

    for root in sorted(modules):
        if color.get(root, 0) == 0:
            _strip(root)

    # Longest path on the resulting DAG (memoizable: no cycles remain).
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
    """Return ``(max_len, chains)`` -- every maximal simple eager chain whose
    length equals the longest. Used for the human-readable recap so a PR that
    deepens the graph shows exactly which chains now sit at the ceiling.
    """
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
        if not extended:  # maximal: cannot extend further
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
    """Human-readable diagnostic: the layer table + direction tally, the measured
    max eager chain vs the ceiling, and every chain tied at the maximum.

    Rendered into the pytest terminal summary by ``test/unit/conftest.py`` so the
    numbers show on every run; the depth gate reuses the chain recap in its
    failure message.
    """
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

    lines = ["Layers (L0 = foundation, ascending to the app entrypoint); "
             "every dependency must point DOWN to a lower or equal layer:"]
    for i, layer in enumerate(LAYERS):
        lines.append(f"  L{i}: " + ", ".join(sorted(layer)))
    lines.append(f"  layered edges: {down} downward (ok), {horiz} horizontal/same-layer, "
                 f"{up} upward (ILLEGAL)")
    lines.append("")
    status = "OK" if max_len <= MAX_CHAIN else "OVER CEILING"
    lines.append(f"Max eager import chain: {max_len} modules "
                 f"(ceiling MAX_CHAIN={MAX_CHAIN}) -> {status}")
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
    """All modules eagerly reachable from ``start`` (direct + indirect)."""
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
