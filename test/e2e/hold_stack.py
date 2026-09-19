# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Boot the end-to-end stack outside pytest and keep it running.

For poking at the same instance the tests see (Navidrome, gunicorn, workers,
an empty or analyzed catalogue) and for running the browser page smoke from
another machine or interpreter through AUDIOMUSE_E2E_BASE_URL. Optionally runs
the shared analysis first. Stops on Ctrl+C, on SIGTERM (kill, pkill) or
after AUDIOMUSE_E2E_HOLD_SECONDS, and always shuts every process down.

Main Features:
* python -m test.e2e.hold_stack [--analyze] boots, prints the URLs and waits
* the same state directory rules as the tests (AUDIOMUSE_E2E_STATE_DIR)
* SIGTERM is turned into the same clean shutdown as Ctrl+C
"""

import os
import signal
import sys
import time

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test.e2e.stack.boot import Stack, require_linux  # noqa: E402
from test.e2e.stack.errors import StackError  # noqa: E402

HOLD_ENV = 'AUDIOMUSE_E2E_HOLD_SECONDS'


def _terminate(_signum, _frame):
    raise KeyboardInterrupt


def main(argv):
    require_linux()
    signal.signal(signal.SIGTERM, _terminate)
    analyze = '--analyze' in argv
    stack = Stack()
    try:
        if not stack.boot():
            print('no database available (set AUDIOMUSE_TEST_DATABASE_URL or install pgserver)')
            return 2
        print(f'flask:     {stack.base_url}')
        print(f'navidrome: {stack.navidrome.base_url}')
        print(f'postgres:  {stack.dsn}')
        print(f'run dir:   {stack.run_dir}')
        if analyze:
            task_id = stack.api.start_task('/api/analysis/start', {'num_recent_albums': 0, 'top_n_moods': 5})
            final = stack.api.wait_for_task(task_id, timeout=900)
            print(f'analysis {task_id}: {final.get("state")} {final.get("status_message")}')
        hold = float(os.environ.get(HOLD_ENV, '0') or 0)
        deadline = time.monotonic() + hold if hold > 0 else None
        print('holding the stack; Ctrl+C to stop' + (f' (auto-stop in {hold:.0f}s)' if deadline else ''))
        while deadline is None or time.monotonic() < deadline:
            problems = stack.alive_problems()
            if problems:
                print('\n'.join(problems))
                return 1
            time.sleep(2)
        return 0
    except StackError as exc:
        print(str(exc))
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stack.shutdown()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
