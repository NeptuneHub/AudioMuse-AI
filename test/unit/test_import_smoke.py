"""Smoke test: every project module must import cleanly.

Complements the flake8 F821 (undefined-name) gate. F821 is static and cannot see
cross-module breakage like ``from x import name_that_no_longer_exists`` or
``import missing_module`` -- actually importing each module catches those plus
any other module-load-time error (a moved function leaving a dangling import, a
typo in a re-export, etc.).

A psycopg2 / redis *connection* error means the import machinery already
succeeded (the module merely tried to reach a service while loading, e.g.
``app`` runs ``init_db()`` at import) and is treated as a pass. Genuinely-missing
optional native deps are skipped; everything else (ImportError, NameError,
SyntaxError, AttributeError) fails the test.
"""
import importlib
import os
from pathlib import Path

import psycopg2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirrors test_import_architecture._collect_modules excludes (plus scripts/ and
# screenshot/, which are standalone tooling entry points -- the latter depends on
# playwright -- not meant to import in a server context).
EXCLUDED_DIRS = {
    ".git", ".venv", ".venv-windows", "node_modules", "__pycache__",
    "build", "dist", "pginstall", "native-build", "test", "scripts", "screenshot",
}

# ImportErrors naming an optional/native dependency that may be absent in CI.
_OPTIONAL_DEPS = ("cuml", "cupy", "ivf", "faiss", "tensorflow")


def _discover_modules():
    modules = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            parts = list((Path(dirpath) / filename).relative_to(REPO_ROOT).parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            if parts:
                modules.append(".".join(parts))
    return sorted(set(modules))


MODULES = _discover_modules()


@pytest.mark.parametrize("modname", MODULES)
def test_module_imports_cleanly(modname):
    """Importing the module must not raise an ImportError/NameError/etc.

    Reaching a DB/Redis connection during import is fine -- it proves the module
    loaded; only a real load-time error should fail.
    """
    try:
        importlib.import_module(modname)
    except psycopg2.OperationalError:
        pytest.skip(f"{modname}: reached DB connection during import (machinery OK)")
    except ImportError as exc:
        msg = str(exc).lower()
        if any(dep in msg for dep in _OPTIONAL_DEPS):
            pytest.skip(f"{modname}: optional dependency absent ({exc})")
        raise
    except Exception as exc:  # noqa: BLE001 -- surface the real failure
        if "redis" in type(exc).__module__ and "connect" in str(exc).lower():
            pytest.skip(f"{modname}: reached Redis connection during import (machinery OK)")
        raise
