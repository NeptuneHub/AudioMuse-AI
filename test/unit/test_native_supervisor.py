"""Unit tests for the native-build ProcessSupervisor concurrency guards.

Covers PR theme #11 (native supervisor stop/boot race) for whichever
per-platform supervisor module imports cleanly on the host running the suite:

  (a) _join_workers skips the CURRENT thread (and, on Windows, the MAIN thread)
      so teardown never blocks on a thread that parks forever.
  (b) _start_health_loop CLEARS the stop event before spawning, so a restart
      after stop_all yields a LIVE health thread instead of one that exits at
      once on a stale set event.
  (c) start_child / start_in_background refuse to spawn once a stop is in
      progress, so a stop racing a boot cannot orphan a child process.

CI runs on Linux; native-build/linux/supervisor.py is the reference and always
imports. macOS/Windows supervisors may pull platform-only deps -- each is loaded
in isolation and SKIPPED cleanly if it cannot import here.
"""
import importlib.util
import os
import sys
import threading
import time

import pytest

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

# Windows is the only supervisor whose _join_workers also skips the main thread
# (its console `start` runs the boot on the main thread, which then parks).
SKIPS_MAIN_THREAD = {'windows'}


def _ensure_path(entry):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load_supervisor(platform_name):
    """Load one platform supervisor in isolation; skip if it cannot import here."""
    _ensure_path(REPO_ROOT)
    _ensure_path(NATIVE_BUILD)
    mod_name = 'native_supervisor_under_test_' + platform_name
    path = os.path.join(NATIVE_BUILD, platform_name, 'supervisor.py')
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        pytest.skip(f"{platform_name} supervisor does not import on this platform: {exc!r}")
    return mod


def _bare_supervisor(mod):
    """Build a ProcessSupervisor without running __init__ (no socket / log file).

    Only the attributes touched by the guards under test are populated.
    """
    sup = mod.ProcessSupervisor.__new__(mod.ProcessSupervisor)
    sup._lock = threading.RLock()
    sup._children = {}
    sup._desired = set()
    sup._state = 'stopped'
    sup._stop_requested = threading.Event()
    sup._health_stop = threading.Event()
    sup._health_thread = None
    sup._boot_thread = None
    sup._database_url = 'postgresql://unused'
    sup._redis_url = 'redis://unused'
    sup._db_conn = 'postgresql://unused'

    class _Log:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    sup._log = _Log()
    return sup


# Parametrize across every supervisor; non-importing ones skip inside the loader.
PLATFORMS = ['linux', 'macos', 'windows']


@pytest.fixture(params=PLATFORMS)
def supervisor_case(request):
    platform_name = request.param
    mod = _load_supervisor(platform_name)
    return platform_name, mod


# ---------------------------------------------------------------------------
# (a) _join_workers thread-skip guards
# ---------------------------------------------------------------------------

class TestJoinWorkersSkips:
    def test_returns_promptly_when_boot_thread_is_current(self, supervisor_case):
        """_join_workers must not block on the calling (current) thread."""
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._boot_thread = threading.current_thread()
        sup._health_thread = None
        start = time.time()
        sup._join_workers()
        assert time.time() - start < 1.0

    def test_alive_non_current_thread_would_be_joined(self, supervisor_case):
        """A sentinel proves a non-current, non-main alive thread IS joined,
        so the current-thread skip is a real skip and not a no-op everywhere."""
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        release = threading.Event()
        joined = threading.Event()

        def _sentinel_body():
            release.wait(2)

        sentinel = threading.Thread(target=_sentinel_body, name='sentinel', daemon=True)
        sentinel.start()
        sup._boot_thread = sentinel
        sup._health_thread = None

        def _runner():
            sup._join_workers()
            joined.set()

        runner = threading.Thread(target=_runner, name='join-runner', daemon=True)
        runner.start()
        # Sentinel is still alive (release not set) -> the joiner must be blocked.
        assert not joined.wait(0.5)
        release.set()
        assert joined.wait(2.0)
        sentinel.join(2)

    def test_skips_main_thread_only_on_windows(self, supervisor_case):
        """Windows boots on the main thread (which then parks forever), so its
        _join_workers must skip the main thread. The other supervisors do not."""
        platform_name, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._boot_thread = threading.main_thread()
        sup._health_thread = None

        done = threading.Event()

        def _runner():
            sup._join_workers()
            done.set()

        runner = threading.Thread(target=_runner, name='main-skip-runner', daemon=True)
        runner.start()

        if platform_name in SKIPS_MAIN_THREAD:
            # Main thread is alive but must be skipped -> returns promptly.
            assert done.wait(1.0)
        else:
            # No main-thread skip: the alive main thread is joined, so the worker
            # parks on the 30s join timeout and does not finish promptly.
            assert not done.wait(0.5)
        runner.join(1)


# ---------------------------------------------------------------------------
# (b) _start_health_loop clears the stop event
# ---------------------------------------------------------------------------

class TestStartHealthLoopClearsStop:
    def test_clears_preset_stop_and_spawns_live_thread(self, supervisor_case, monkeypatch):
        platform_name, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._state = 'running'

        body_ran = threading.Event()

        # Make the loop body harmless and observable, and make the wait
        # non-blocking so the spawned thread iterates quickly regardless of the
        # platform's wait interval.
        def _record_and_stop(*_a, **_k):
            body_ran.set()
            sup._health_stop.set()

        # Linux/macOS health loop calls these infra checks each iteration.
        for name in ('_ensure_postgres_healthy', '_ensure_redis_healthy'):
            if hasattr(sup, name):
                monkeypatch.setattr(sup, name, _record_and_stop)
        # Windows loop body issues an HTTP poll; neutralize and observe it.
        if hasattr(mod, 'urllib'):
            monkeypatch.setattr(mod.urllib.request, 'urlopen', _record_and_stop)

        # Pre-set the stop event: a stale set event is exactly the restart bug.
        sup._health_stop.set()
        assert sup._health_stop.is_set()

        real_event = sup._health_stop
        non_blocking_wait = lambda timeout=None: real_event.is_set()
        monkeypatch.setattr(real_event, 'wait', non_blocking_wait)

        try:
            sup._start_health_loop()
            # The guard under test: the stop event was cleared before spawning.
            # (The loop body re-sets it via _record_and_stop once it runs.)
            assert sup._health_thread is not None
            assert sup._health_thread.is_alive() or body_ran.is_set()
            assert body_ran.wait(2.0), "spawned health loop body never executed"
        finally:
            real_event.set()
            if sup._health_thread is not None:
                sup._health_thread.join(2)


# ---------------------------------------------------------------------------
# (c) spawn refused once a stop is in progress
# ---------------------------------------------------------------------------

class TestSpawnRefusedWhileStopping:
    def test_start_child_refuses_when_stopping(self, supervisor_case, monkeypatch):
        """A stop racing a boot must not create a child: with state == stopping,
        start_child returns False and never calls subprocess.Popen."""
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)

        popen_calls = []

        def _fake_popen(*a, **k):
            popen_calls.append((a, k))
            raise AssertionError("Popen must not be called once stopping")

        monkeypatch.setattr(mod.subprocess, 'Popen', _fake_popen)

        sup._state = 'stopping'
        sup._stop_requested.set()

        result = sup.start_child('flask')
        assert result is False
        assert popen_calls == []
        assert 'flask' not in sup._children
        assert 'flask' not in sup._desired

    def test_start_child_allowed_while_starting(self, supervisor_case, monkeypatch):
        """Sanity: the guard is state-specific -- in the 'starting' state the
        child IS marked desired (proving the refusal is the stop, not a blanket
        block). Popen is mocked so no real process spawns."""
        platform_name, mod = supervisor_case
        sup = _bare_supervisor(mod)

        class _FakePopen:
            def __init__(self, *a, **k):
                self.pid = 4321
                self.stdout = None

            def poll(self):
                return None

        monkeypatch.setattr(mod.subprocess, 'Popen', _FakePopen)
        # Neutralize platform side effects reached only after the guard passes.
        for name in ('_terminate_named',):
            if hasattr(sup, name):
                monkeypatch.setattr(sup, name, lambda *a, **k: None)
        if hasattr(mod, 'threading'):
            monkeypatch.setattr(mod.threading, 'Thread', lambda *a, **k: type(
                'T', (), {'start': lambda self: None, 'daemon': True})())
        # Windows start_child touches db_backend / env / redis before spawning.
        if hasattr(mod, 'db_backend'):
            monkeypatch.setattr(mod.db_backend, 'ensure_embedded_running', lambda *a, **k: 'postgresql://x')
        if hasattr(mod, 'env_builder'):
            monkeypatch.setattr(mod.env_builder, 'build_child_env', lambda *a, **k: {})
        if hasattr(sup, '_ensure_redis_running'):
            monkeypatch.setattr(sup, '_ensure_redis_running', lambda *a, **k: 'redis://x')

        sup._state = 'starting'
        result = sup.start_child('flask')
        assert result is True
        assert 'flask' in sup._desired


# ---------------------------------------------------------------------------
# start_in_background owns the boot thread (Linux/macOS only)
# ---------------------------------------------------------------------------

class TestStartInBackground:
    def test_owns_boot_thread_and_invokes_start_all(self, supervisor_case, monkeypatch):
        platform_name, mod = supervisor_case
        if not hasattr(mod.ProcessSupervisor, 'start_in_background'):
            pytest.skip(f"{platform_name} supervisor has no start_in_background")

        sup = _bare_supervisor(mod)
        started = threading.Event()
        boot_thread_seen = {}

        def _fake_start_all():
            boot_thread_seen['thread'] = threading.current_thread()
            started.set()

        monkeypatch.setattr(sup, 'start_all', _fake_start_all)
        # is_running() gates the on_ready callback; keep it False to avoid it.
        monkeypatch.setattr(sup, 'is_running', lambda: False)

        thread = sup.start_in_background()
        assert thread is sup._boot_thread
        assert started.wait(2.0)
        # start_all ran on the supervisor-owned background thread, not the caller.
        assert boot_thread_seen['thread'] is sup._boot_thread
        assert boot_thread_seen['thread'] is not threading.current_thread()
        thread.join(2)
