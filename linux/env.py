"""Build the environment handed to each supervised child process (Linux build).

Linux counterpart of ``macos/env.py``. The standalone-mode overrides are
centralized here so every child imports ``config`` already pointed at the
embedded services and the bundled models. Keys mirror the env-driven settings in
``config.py`` (``DATABASE_URL``, ``REDIS_URL``, ``TEMP_DIR``, the ``*_MODEL_PATH``
values, ``LYRICS_MODEL_DIR``) plus the runtime signals the supervisor and
``restart_manager`` agree on (``AUDIOMUSE_PLATFORM``, control socket, roles).

Why ``AUDIOMUSE_PLATFORM=macos`` on Linux
-----------------------------------------
``restart_manager.py`` (shared code we must not modify) has exactly one
platform-keyed branch: ``if config.AUDIOMUSE_PLATFORM == 'macos'`` it forwards
"restart workers"/"stop flask" requests to a unix-socket *control server*
instead of shelling out to ``supervisorctl`` (which only exists in the
container). That control-server protocol is platform-agnostic, and this build's
supervisor implements it identically (via ``macos.control_ipc.ControlServer``).
So reporting ``macos`` here makes the web UI's "save config -> restart workers"
flow work on the native Linux build with **zero shared-code changes**. The value
is an internal "standalone embedded supervisor" signal, not a real OS check.

Note the macOS-only workarounds from ``macos/env.py`` are intentionally omitted:
``OBJC_DISABLE_INITIALIZE_FORK_SAFETY`` and the ``LC_NUMERIC=C`` pin both address
Apple-framework behavior that does not exist on Linux (and Linux is the platform
``config.py``/the container already target, so its defaults are correct here).
"""

import contextlib
import os
import sys

from linux import paths

_WORKER_ROLES = {"worker-high", "worker-default", "janitor", "restart-listener"}


def restore_native_lib_path(env):
    """Undo PyInstaller's ``LD_LIBRARY_PATH`` injection in ``env`` (mutates+returns it).

    PyInstaller's bootloader prepends the frozen bundle's ``_internal`` dir to
    ``LD_LIBRARY_PATH`` so the frozen Python finds its own bundled ``.so`` files,
    saving the prior value in ``LD_LIBRARY_PATH_ORIG``. A *bundled external*
    binary (``initdb``/``postgres`` via pgserver, ``redis-server``) that inherits
    this polluted path loads the bundle's incompatible ``libssl``/``libz``/
    ``libstdc++`` instead of its own (rpath ``$ORIGIN/../lib``) libraries and
    crashes -- pgserver's ``initdb`` dies with SIGSEGV. Restore the original
    value so the child resolves its own libraries (PyInstaller documents this as
    the required handling before executing other programs).

    No-op outside the frozen build (``LD_LIBRARY_PATH_ORIG`` is set only by the
    bootloader), so a developer's own ``LD_LIBRARY_PATH`` is left untouched.
    """
    if not getattr(sys, "frozen", False):
        return env
    orig = env.get("LD_LIBRARY_PATH_ORIG")
    if orig:
        env["LD_LIBRARY_PATH"] = orig
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


@contextlib.contextmanager
def native_lib_path_restored():
    """Temporarily apply :func:`restore_native_lib_path` to ``os.environ``.

    For native binaries spawned by a library we don't control: pgserver runs
    ``initdb``/``postgres`` with an inherited ``os.environ`` we can't override per
    call. We narrow the scrub to the spawn window and restore ``os.environ``
    afterwards so the frozen parent's own later ``dlopen`` calls keep resolving
    the bundled libraries.
    """
    if not getattr(sys, "frozen", False):
        yield
        return
    saved = os.environ.get("LD_LIBRARY_PATH")
    try:
        restore_native_lib_path(os.environ)
        yield
    finally:
        if saved is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = saved


def build_child_env(role, database_url, redis_url):
    """Return an ``os.environ`` copy with the embedded-mode overrides for ``role``."""
    env = dict(os.environ)
    model_dir = paths.model_dir()
    env.update({
        # See module docstring: selects the control-socket restart path in the
        # shared restart_manager.py. Not a real OS check.
        "AUDIOMUSE_PLATFORM": "macos",
        "APP_DATA_DIR": paths.app_support_dir(),
        "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
        "DATABASE_TYPE": "embedded",
        "QUEUE_TYPE": "embedded",
        "DATABASE_URL": database_url,
        "REDIS_URL": redis_url,
        "TEMP_DIR": paths.temp_audio_dir(),
        "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
        # transformers/huggingface_hub look up models (e.g. the CLAP RoBERTa
        # tokenizer, loaded with local_files_only=True) in HF_HOME/hub. The
        # container points HF_HOME at a pre-baked cache; we bundle the same cache
        # at model/huggingface and point there. Offline flags guarantee no
        # network stall on a machine that may be offline -- the bundled cache is
        # complete.
        "HF_HOME": os.path.join(model_dir, "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EMBEDDING_MODEL_PATH": os.path.join(model_dir, "musicnn_embedding.onnx"),
        "PREDICTION_MODEL_PATH": os.path.join(model_dir, "musicnn_prediction.onnx"),
        "CLAP_AUDIO_MODEL_PATH": os.path.join(model_dir, "model_epoch_36.onnx"),
        "CLAP_TEXT_MODEL_PATH": os.path.join(model_dir, "clap_text_model.onnx"),
        "LYRICS_MODEL_DIR": model_dir,
        # The lyrics ONNX loaders (lyrics/silero_onnx.py, lyrics/gte_onnx.py)
        # hardcode container ``/app/model/...`` defaults and do NOT derive from
        # LYRICS_MODEL_DIR, so each needs its own override or the frozen app
        # looks at the container path. (Whisper does derive from
        # LYRICS_MODEL_DIR, but we set it explicitly too for clarity.)
        "LYRICS_WHISPER_MODEL_DIR": os.path.join(model_dir, "whisper-small-onnx"),
        "SILERO_VAD_ONNX_PATH": os.path.join(model_dir, "silero_vad.onnx"),
        "LYRICS_GTE_ONNX_PATH": os.path.join(model_dir, "gte-multilingual-base-int8.onnx"),
        "LYRICS_GTE_TOKENIZER_DIR": os.path.join(model_dir, "gte-multilingual-base"),
        # Backup/restore (app_backup.py). Its defaults are container-shaped: it
        # writes pg_dump output under /app/backup (read-only in the bundle) and
        # builds pg_dump/psql/pg_restore connection args from the POSTGRES_* vars
        # (TCP host/port that don't match the embedded server). Repoint all of it
        # at the embedded unix socket + the writable data dir. Only app_backup
        # reads these POSTGRES_* values; the app itself connects via DATABASE_URL,
        # which the supervisor already sets, so this does not affect anything else.
        "BACKUP_DIR": paths.backup_dir(),
        "RESTORE_LOG_DIR": paths.backup_dir(),
        "POSTGRES_HOST": paths.pgdata_dir(),  # libpq treats a /path host as a unix-socket dir
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "",
        "POSTGRES_DB": "postgres",
        # Put the bundled Postgres client tools on PATH so pg_dump et al.
        # resolve. Build with a filtered join so an unset/empty inherited PATH
        # cannot leave a trailing separator -- an empty PATH entry is treated as
        # the current working directory on Unix (CWE-426, untrusted search path).
        "PATH": os.pathsep.join(filter(None, [paths.pg_bin_dir(), os.environ.get("PATH")])),
    })
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
