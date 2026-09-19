#!/usr/bin/env bash
# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
#
# Runs the end-to-end suite on a developer machine (Linux or WSL) against the
# repo .venv. Postgres comes from AUDIOMUSE_TEST_DATABASE_URL or pgserver, the
# models from model/ (or AUDIOMUSE_E2E_MODEL_DIR), Navidrome and fpcalc are
# downloaded once into test/.cache. Pass --no-browser to leave out the
# Playwright page smoke (Chromium needs system libraries that WSL usually lacks).
# Every other argument goes to pytest, for example -k cold or -x.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "run this inside WSL or Linux; the end-to-end stack is Linux only" >&2
  exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    echo "no .venv in $REPO_ROOT; create it and pip install -r test/requirements.txt" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

MARKER=()
PYTEST_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-browser) MARKER=(-m "not browser") ;;
    *) PYTEST_ARGS+=("$arg") ;;
  esac
done

python - <<'PY'
import importlib.util
import sys

needed = ("pytest_timeout", "gunicorn", "psycopg2", "av", "psutil", "requests", "langdetect")
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing packages: " + ", ".join(missing) + " (pip install -r test/requirements.txt)")
PY

if [[ -z "${AUDIOMUSE_TEST_DATABASE_URL:-}" ]]; then
  if ! python -c "import pgserver" 2>/dev/null; then
    echo "set AUDIOMUSE_TEST_DATABASE_URL to a disposable database or pip install pgserver==0.1.4" >&2
    exit 2
  fi
  # pgserver ships a minimal PostgreSQL without the unaccent and pg_trgm
  # contrib modules that init_db creates. The Linux native build compiles them
  # against pgserver's own headers; reuse that script once and graft the
  # artifacts into pgserver's install tree, exactly like the PyInstaller spec.
  PGINSTALL="$(python -c 'import os, pgserver; print(os.path.join(os.path.dirname(pgserver.__file__), "pginstall"))')"
  if [[ ! -f "$PGINSTALL/share/postgresql/extension/unaccent.control" ]]; then
    CONTRIB="native-build/linux/vendor/pg-contrib/$(uname -m)"
    if [[ ! -f "$CONTRIB/extension/unaccent.control" ]]; then
      echo "building the unaccent and pg_trgm extensions against pgserver's PostgreSQL (needs gcc, make, curl)"
      bash native-build/linux/vendor/pg-contrib/build-pg-contrib.sh
    fi
    cp "$CONTRIB"/lib/*.so "$PGINSTALL/lib/postgresql/"
    cp "$CONTRIB"/extension/* "$PGINSTALL/share/postgresql/extension/"
    mkdir -p "$PGINSTALL/share/postgresql/tsearch_data"
    cp "$CONTRIB"/tsearch_data/* "$PGINSTALL/share/postgresql/tsearch_data/"
    echo "installed unaccent and pg_trgm into $PGINSTALL"
  fi
fi

export PYTHONUNBUFFERED=1
python -m pytest test/e2e/ -p no:cacheprovider --timeout=2700 --timeout-method=signal \
  "${MARKER[@]}" "${PYTEST_ARGS[@]}"
echo "run artifacts: $REPO_ROOT/test/e2e/.run"
