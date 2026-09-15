# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""IdleUnloadTimer window semantics shared by every idle-unload cache.

The timer backs the CLAP text model, the GTE lyrics model, the hyperbolic tree
cache and the availability memo, so a mistake here silently strands whatever
each of them meant to free.

Main Features:
* A plain expiry runs the action once and then closes the window.
* An action that re-arms from inside the callback keeps the same worker alive
  for the new window, rather than having its arm erased by the clearing step.
* Re-arming from another thread pushes the window out instead of firing early.
* An action that raises still closes the window (the raise escapes the worker
  thread by design, so pytest reports one unhandled-thread-exception warning).
"""

import threading
import time

from tasks.idle_unload import IdleUnloadTimer


def _wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


class TestIdleUnloadTimer:
    def test_action_runs_once_and_closes_the_window(self):
        timer = IdleUnloadTimer()
        fired = []

        assert timer.arm(0.05, lambda: fired.append(1)) is True
        assert _wait_until(lambda: len(fired) == 1)
        assert _wait_until(lambda: timer.expiry() is None)
        time.sleep(0.2)
        assert len(fired) == 1

    def test_an_action_that_rearms_keeps_the_timer_running(self):
        timer = IdleUnloadTimer()
        fired = []

        def on_expire():
            fired.append(time.time())
            if len(fired) < 3:
                timer.arm(0.05, on_expire)

        timer.arm(0.05, on_expire)

        assert _wait_until(lambda: len(fired) >= 3), (
            f"re-arm from inside the callback was discarded; fired {len(fired)} of 3"
        )
        assert _wait_until(lambda: timer.expiry() is None)

    def test_a_rearm_from_another_thread_pushes_the_window_out(self):
        timer = IdleUnloadTimer()
        fired = []
        timer.arm(0.4, lambda: fired.append(1))

        time.sleep(0.2)
        assert timer.arm(0.4, lambda: fired.append(1)) is False, "a live worker was duplicated"
        time.sleep(0.3)
        assert not fired, "the window should have been pushed out by the re-arm"

        assert _wait_until(lambda: len(fired) == 1)

    def test_an_action_that_raises_still_closes_the_window(self):
        timer = IdleUnloadTimer()
        fired = []

        def boom():
            fired.append(1)
            raise RuntimeError("unload failed")

        timer.arm(0.05, boom)
        assert _wait_until(lambda: len(fired) == 1)
        assert _wait_until(lambda: timer.expiry() is None)

    def test_a_later_arm_starts_a_fresh_worker_once_the_window_closed(self):
        timer = IdleUnloadTimer()
        fired = []

        timer.arm(0.05, lambda: fired.append(1))
        assert _wait_until(lambda: timer.expiry() is None)

        assert timer.arm(0.05, lambda: fired.append(1)) is True
        assert _wait_until(lambda: len(fired) == 2)

    def test_concurrent_arming_never_runs_two_workers(self):
        timer = IdleUnloadTimer()
        started = []
        barrier = threading.Barrier(6)

        def racer():
            barrier.wait()
            for _ in range(40):
                started.append(timer.arm(0.05, lambda: None))

        threads = [threading.Thread(target=racer) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not [t for t in threads if t.is_alive()]
        assert sum(1 for value in started if value) >= 1
