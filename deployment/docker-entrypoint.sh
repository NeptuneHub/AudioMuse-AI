#!/usr/bin/env bash
set -euo pipefail

echo "ENTRYPOINT: SERVICE_TYPE=${SERVICE_TYPE:-<unset>}"

if [ -n "${TZ:-}" ]; then
  if [ -f "/usr/share/zoneinfo/$TZ" ]; then
    ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
    echo "$TZ" > /etc/timezone
  else
    echo "Warning: timezone '$TZ' not found in /usr/share/zoneinfo" >&2
  fi
fi

if [ ! -f /etc/supervisor/conf.d/supervisord.conf ]; then
  echo "ERROR: supervisor config file missing: /etc/supervisor/conf.d/supervisord.conf" >&2
  exit 1
fi

exec_supervisorctl() {
  echo "SUPERVISORCTL: $*"
  /usr/bin/supervisorctl -c /etc/supervisor/conf.d/supervisord.conf "$@"
}

/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &
SUPERVISORD_PID=$!

shutdown_supervisord() {
  local signal="${1:-TERM}"
  echo "ENTRYPOINT: received SIG${signal}, forwarding to supervisord pid ${SUPERVISORD_PID}"

  set +e
  if [ -n "${SUPERVISORD_PID:-}" ] && kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
    kill -"$signal" "$SUPERVISORD_PID" 2>/dev/null
    wait "$SUPERVISORD_PID"
    local exit_code=$?
    echo "ENTRYPOINT: supervisord stopped with code ${exit_code}"
    exit "$exit_code"
  fi
  echo "ENTRYPOINT: supervisord pid is not running"
  exit 0
}

trap 'shutdown_supervisord TERM' TERM
trap 'shutdown_supervisord INT' INT

SUPERVISOR_SOCK=/tmp/supervisor.sock
while true; do
  if [ -S "$SUPERVISOR_SOCK" ]; then
    echo "ENTRYPOINT: supervisord socket ready"
    break
  fi
  if ! kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
    echo "supervisord exited unexpectedly" >&2
    exit 1
  fi
  echo "ENTRYPOINT: waiting for supervisord socket"
  sleep 0.25
done

echo "ENTRYPOINT: supervisord RPC is ready"
exec_supervisorctl avail

echo "ENTRYPOINT: ready to start ${SERVICE_TYPE:-flask}"

if [ "${SERVICE_TYPE:-flask}" = "worker" ]; then
  echo 'Starting worker processes via supervisord...'
  set +e
  START_OUTPUT=$(exec_supervisorctl start rq-worker-default rq-worker-high rq-janitor 2>&1)
  START_CODE=$?
  set -e
  echo "SUPERVISORCTL START OUTPUT: $START_OUTPUT"
  echo "SUPERVISORCTL START EXIT CODE: $START_CODE"
  if [ "$START_CODE" -ne 0 ]; then
    echo "Supervisor start failed for worker processes" >&2
    exit "$START_CODE"
  fi
else
  echo 'Starting web service via supervisord...'
  set +e
  START_OUTPUT=$(exec_supervisorctl start flask 2>&1)
  START_CODE=$?
  set -e
  echo "SUPERVISORCTL START OUTPUT: $START_OUTPUT"
  echo "SUPERVISORCTL START EXIT CODE: $START_CODE"
  if [ "$START_CODE" -ne 0 ]; then
    echo "Supervisor start failed for flask" >&2
    exit "$START_CODE"
  fi
fi

echo "SUPERVISORCTL STATUS AFTER START:"
set +e
exec_supervisorctl status
STATUS_EXIT_CODE=$?
echo "SUPERVISORCTL STATUS EXIT CODE: $STATUS_EXIT_CODE"
exec_supervisorctl avail
AVAIL_EXIT_CODE=$?
echo "SUPERVISORCTL AVAIL EXIT CODE: $AVAIL_EXIT_CODE"
set -e

echo "ENTRYPOINT: now waiting on supervisord pid $SUPERVISORD_PID"
wait "$SUPERVISORD_PID"
SUPERVISORD_EXIT_CODE=$?
echo "ENTRYPOINT: supervisord pid $SUPERVISORD_PID exited with code $SUPERVISORD_EXIT_CODE"
exit "$SUPERVISORD_EXIT_CODE"
