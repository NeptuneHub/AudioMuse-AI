#!/usr/bin/env bash
set -euo pipefail

echo "ENTRYPOINT: SERVICE_TYPE=${SERVICE_TYPE:-<unset>}"

if [ -n "$TZ" ]; then
  if [ -f "/usr/share/zoneinfo/$TZ" ]; then
    ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
    echo "$TZ" > /etc/timezone
  else
    echo "Warning: timezone '$TZ' not found in /usr/share/zoneinfo" >&2
  fi
fi

exec_supervisorctl() {
  echo "SUPERVISORCTL: $*"
  /usr/bin/supervisorctl -c /etc/supervisor/conf.d/supervisord.conf "$@"
}

/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &
SUPERVISORD_PID=$!

while true; do
  set +e
  STATUS_OUTPUT=$(exec_supervisorctl status 2>&1)
  STATUS_CODE=$?
  set -e
  if echo "$STATUS_OUTPUT" | grep -qE '^(config-restart-listener|flask|rq-janitor|rq-worker-default|rq-worker-high)\s+'; then
    echo "ENTRYPOINT: supervisord RPC ready"
    break
  fi
  echo "ENTRYPOINT: waiting for supervisord RPC"
  echo "ENTRYPOINT: supervisorctl status exit code: $STATUS_CODE"
  echo "ENTRYPOINT: supervisorctl status output: $STATUS_OUTPUT"
  sleep 0.5
  if ! kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
    echo "supervisord exited unexpectedly" >&2
    exit 1
  fi
done

echo "ENTRYPOINT: supervisord RPC is ready"
exec_supervisorctl avail

echo "ENTRYPOINT: ready to start $SERVICE_TYPE"

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
exec_supervisorctl status

echo "SUPERVISORCTL SERVERS AFTER START:"
exec_supervisorctl avail

wait "$SUPERVISORD_PID"
