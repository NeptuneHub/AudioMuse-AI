#!/usr/bin/env bash
# Wipe an AudioMuse-AI docker-compose test stack and re-register its media
# servers, so a provider test can be repeated from a known-empty database
# without clicking through the setup wizard every time.
#
# Run it FROM (or pointed AT) the directory holding docker-compose.yaml.
#
#   ./ampache-reset.sh reset            # stop, delete DB/redis/temp, start clean
#   ./ampache-reset.sh restore          # register the servers in servers.json
#   ./ampache-reset.sh full             # reset + restore
#   ./ampache-reset.sh rebuild          # git pull + docker build, then full
#   ./ampache-reset.sh status           # what is registered right now
#
# Add --yes to skip the "this deletes your database" confirmation.
#
# servers.json (kept next to this script, NOT in git - it holds credentials):
#
#   [
#     {"name": "Ampache",   "server_type": "ampache",
#      "creds": {"url": "https://ampache.local", "user": "amp", "password": "KEY-OR-PASSWORD"},
#      "music_libraries": "", "make_default": true},
#     {"name": "Navidrome", "server_type": "navidrome",
#      "creds": {"url": "http://navidrome:4533", "user": "admin", "password": "secret"},
#      "music_libraries": ""}
#   ]
#
# Registration is idempotent: a server whose name already exists is skipped, so
# `restore` is safe to re-run and safe to use after an env-bootstrapped boot has
# already seeded the default server.
#
# For a test stack, disable the login layer so neither the UI nor this script
# needs credentials - with auth off every caller is treated as admin. Put this in
# the stack's .env AND pass it through in the compose environment of both the
# flask and worker services:
#
#   AUTH_ENABLED=false
#
# With AUTH_ENABLED=false nothing else is needed. If you would rather keep auth
# on, set API_TOKEN=<secret> instead (plus AUDIOMUSE_USER / AUDIOMUSE_PASSWORD so
# the wizard can be skipped) and the script will send it as a Bearer token.
#
# Requires: docker compose, curl, python3 (stdlib only).

set -euo pipefail

STACK_DIR="${STACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SRC_DIR="${SRC_DIR:-$STACK_DIR/src}"
SERVERS_FILE="${SERVERS_FILE:-$STACK_DIR/servers.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
IMAGE_TAG="${IMAGE_TAG:-audiomuse-ai:ampache}"
WORKERS="${WORKERS:-}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
# Directories wiped by `reset`. Plugins are deliberately NOT here: wiping them
# forces every plugin's pip dependencies to be downloaded again.
WIPE_PATHS=(data/postgres data/redis data/temp-audio-flask data/temp-audio-worker)

ASSUME_YES=0
MODE=""
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    reset|restore|full|rebuild|status) MODE="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 2; }

cd "$STACK_DIR"
[ -f docker-compose.yaml ] || [ -f docker-compose.yml ] || {
  echo "No docker-compose.yaml in $STACK_DIR (set STACK_DIR=...)" >&2; exit 1
}

# Settings are read from the environment first, then from the stack's .env, so
# credentials live in one place. Two of them decide how `restore` talks to the API:
#   AUTH_ENABLED=false -> every caller is admin, no token needed (test stacks)
#   API_TOKEN=<secret> -> authenticates as an admin-equivalent M2M caller
dotenv_value() {
  [ -f .env ] || return 0
  sed -n "s/^[[:space:]]*$1=//p" .env | tail -1 | tr -d '"'"'"'\r'
}

[ -n "${API_TOKEN:-}" ] || API_TOKEN="$(dotenv_value API_TOKEN)"
[ -n "${AUTH_ENABLED:-}" ] || AUTH_ENABLED="$(dotenv_value AUTH_ENABLED)"
API_TOKEN="${API_TOKEN:-}"
AUTH_ENABLED="${AUTH_ENABLED:-true}"

auth_disabled() {
  case "$(printf '%s' "$AUTH_ENABLED" | tr '[:upper:]' '[:lower:]')" in
    false|0|no) return 0 ;;
    *) return 1 ;;
  esac
}

log() { printf '\n=== %s\n' "$*"; }

compose_up() {
  if [ -n "$WORKERS" ]; then
    docker compose up -d --scale "audiomuse-ai-worker=$WORKERS"
  else
    docker compose up -d
  fi
}

confirm_destructive() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  echo "About to DELETE, under $STACK_DIR:"
  printf '  %s\n' "${WIPE_PATHS[@]}"
  echo "This destroys the analysis database. Type 'yes' to continue:"
  read -r reply
  [ "$reply" = "yes" ] || { echo "aborted"; exit 1; }
}

do_reset() {
  confirm_destructive
  log "Stopping the stack"
  docker compose down
  log "Deleting stack data"
  for path in "${WIPE_PATHS[@]}"; do
    [ -e "$path" ] || continue
    # sudo only if the current user cannot remove it (postgres files are root-owned)
    rm -rf "$path" 2>/dev/null || sudo rm -rf "$path"
    echo "  removed $path"
  done
  log "Starting the stack"
  compose_up
  wait_ready
}

wait_ready() {
  log "Waiting for $BASE_URL/api/health (timeout ${READY_TIMEOUT}s)"
  local waited=0
  until curl -fsS --max-time 5 "$BASE_URL/api/health" >/dev/null 2>&1; do
    waited=$((waited + 3))
    if [ "$waited" -ge "$READY_TIMEOUT" ]; then
      echo "Timed out. Last 40 log lines:" >&2
      docker compose logs --tail 40 audiomuse-ai-flask >&2
      exit 1
    fi
    sleep 3
  done
  echo "  ready after ${waited}s"
}

# Registration and status both run through python3: it speaks JSON without
# needing jq, and it can report per-server results instead of a raw HTTP body.
run_python() {
  BASE_URL="$BASE_URL" API_TOKEN="$API_TOKEN" SERVERS_FILE="$SERVERS_FILE" \
    python3 - "$1" <<'PYTHON'
import json
import os
import sys
import urllib.error
import urllib.request

base = os.environ['BASE_URL'].rstrip('/')
token = os.environ.get('API_TOKEN') or ''
mode = sys.argv[1]


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    request.add_header('Content-Type', 'application/json')
    if token:
        request.add_header('Authorization', 'Bearer ' + token)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode() or '{}'
            return response.status, json.loads(body)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {'error': raw[:400]}


def existing():
    status, body = call('GET', '/api/servers')
    if status != 200:
        sys.exit(f'GET /api/servers failed ({status}): {body}')
    servers = body.get('servers') or []
    return body, {str(s.get('name', '')).lower(): s for s in servers}


def show():
    body, by_name = existing()
    default_id = body.get('default_server_id') or body.get('default_id')
    if not by_name:
        print('  (no servers registered)')
    for server in by_name.values():
        mark = ' [default]' if server.get('is_default') or server.get('server_id') == default_id else ''
        print(f"  {server.get('name')}  type={server.get('server_type')}"
              f"  libraries={server.get('music_libraries') or '(all)'}{mark}")


if mode == 'status':
    show()
    sys.exit(0)

path = os.environ['SERVERS_FILE']
if not os.path.exists(path):
    sys.exit(f'{path} not found - create it (see the header of ampache-reset.sh)')
with open(path, encoding='utf-8') as handle:
    wanted = json.load(handle)
if not isinstance(wanted, list):
    sys.exit('servers.json must be a JSON array of server objects')

_body, by_name = existing()
failures = 0
for entry in wanted:
    name = (entry.get('name') or '').strip()
    if not name:
        print('  SKIP: an entry has no name')
        failures += 1
        continue
    if name.lower() in by_name:
        print(f'  SKIP {name}: already registered')
        continue
    payload = {
        'name': name,
        'server_type': entry.get('server_type'),
        'creds': entry.get('creds') or {},
        'music_libraries': entry.get('music_libraries') or '',
        'make_default': bool(entry.get('make_default')),
    }
    status, body = call('POST', '/api/servers', payload)
    if status == 201:
        print(f"  ADDED {name} ({payload['server_type']}), sweep={body.get('sweep_task_id')}")
    else:
        print(f'  FAILED {name} ({status}): {body.get("error") or body}')
        failures += 1

print('\nRegistered now:')
show()
sys.exit(1 if failures else 0)
PYTHON
}

do_restore() {
  wait_ready
  log "Registering servers from $SERVERS_FILE"
  if auth_disabled; then
    echo "  AUTH_ENABLED=false: calling the API unauthenticated (every caller is admin)."
  elif [ -n "$API_TOKEN" ]; then
    echo "  Authenticating with API_TOKEN as an admin-equivalent M2M caller."
  else
    echo "  WARNING: auth is enabled and no API_TOKEN was found. Registration will"
    echo "           only work while the setup wizard is still in progress. Either set"
    echo "           AUTH_ENABLED=false (test stacks) or API_TOKEN=<secret> in .env,"
    echo "           and make sure the flask service passes it through."
  fi
  run_python register
}

do_rebuild() {
  log "Updating the source checkout in $SRC_DIR"
  git -C "$SRC_DIR" fetch --all --prune
  git -C "$SRC_DIR" pull --ff-only
  git -C "$SRC_DIR" log --oneline -1
  log "Building $IMAGE_TAG"
  docker build -t "$IMAGE_TAG" "$SRC_DIR"
}

case "$MODE" in
  reset)   do_reset ;;
  restore) do_restore ;;
  full)    do_reset; do_restore ;;
  rebuild) do_rebuild; do_reset; do_restore ;;
  status)  run_python status ;;
esac

log "Done ($MODE)"
