#!/usr/bin/env bash
set -euo pipefail

SUPERVISOR_CONF=/etc/supervisor/conf.d/supervisord.conf
SUPERVISOR_SOCK=/tmp/supervisor.sock
SERVICE_TYPE="${SERVICE_TYPE:-flask}"

log() {
  echo "ENTRYPOINT: $*"
}

run_supervisorctl() {
  log "SUPERVISORCTL: $*"
  /usr/bin/supervisorctl -c "$SUPERVISOR_CONF" "$@"
}

run_supervisorctl_checked() {
  local output
  local code

  set +e
  output=$(run_supervisorctl "$@" 2>&1)
  code=$?
  set -e

  log "SUPERVISORCTL OUTPUT: $output"
  log "SUPERVISORCTL EXIT CODE: $code"

  if [ "$code" -ne 0 ]; then
    echo "Supervisor command failed: $*" >&2
    exit "$code"
  fi
}

log "SERVICE_TYPE=${SERVICE_TYPE}"

if [ -n "${TZ:-}" ] && [ ! -f "/usr/share/zoneinfo/$TZ" ]; then
  echo "Warning: timezone '$TZ' not found in /usr/share/zoneinfo" >&2
fi

if [ ! -f "$SUPERVISOR_CONF" ]; then
  echo "ERROR: supervisor config file missing: $SUPERVISOR_CONF" >&2
  exit 1
fi

/usr/bin/supervisord -c "$SUPERVISOR_CONF" &
SUPERVISORD_PID=$!

shutdown_supervisord() {
  local signal="${1:-TERM}"
  trap - TERM INT
  log "received SIG${signal}, forwarding to supervisord pid ${SUPERVISORD_PID}"

  set +e
  if [ -n "${SUPERVISORD_PID:-}" ] && kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
    kill -"$signal" "$SUPERVISORD_PID" 2>/dev/null
    wait "$SUPERVISORD_PID"
    local exit_code=$?
    log "supervisord stopped with code ${exit_code}"
    exit "$exit_code"
  fi
  log "supervisord pid is not running"
  exit 0
}

trap 'shutdown_supervisord TERM' TERM
trap 'shutdown_supervisord INT' INT

for _ in $(seq 1 120); do
  if [ -S "$SUPERVISOR_SOCK" ]; then
    log "supervisord socket ready"
    break
  fi
  if ! kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
    echo "supervisord exited unexpectedly" >&2
    exit 1
  fi
  log "waiting for supervisord socket"
  sleep 0.25
done

if [ ! -S "$SUPERVISOR_SOCK" ]; then
  echo "timed out waiting for supervisord socket" >&2
  shutdown_supervisord TERM
fi

log "supervisord RPC is ready"
run_supervisorctl_checked avail

log "ready to start ${SERVICE_TYPE}"

case "$SERVICE_TYPE" in
  worker)
    log "starting worker processes via supervisord"
    run_supervisorctl_checked start rq-worker-default rq-worker-high rq-janitor
    ;;
  flask)
    log "starting web service via supervisord"
    run_supervisorctl_checked start flask
    ;;
  *)
    echo "Unsupported SERVICE_TYPE '$SERVICE_TYPE' (expected: flask or worker)" >&2
    shutdown_supervisord TERM
    ;;
esac

log "SUPERVISORCTL STATUS AFTER START"
set +e
run_supervisorctl status
STATUS_EXIT_CODE=$?
log "SUPERVISORCTL STATUS EXIT CODE: $STATUS_EXIT_CODE"
run_supervisorctl avail
AVAIL_EXIT_CODE=$?
log "SUPERVISORCTL AVAIL EXIT CODE: $AVAIL_EXIT_CODE"
set -e

log "now waiting on supervisord pid $SUPERVISORD_PID"
wait "$SUPERVISORD_PID"
SUPERVISORD_EXIT_CODE=$?
log "supervisord pid $SUPERVISORD_PID exited with code $SUPERVISORD_EXIT_CODE"
exit "$SUPERVISORD_EXIT_CODE"
