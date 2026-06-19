"""Work around the macOS-arm64 NumPy int->longdouble crash for the native app.

Symptom (issue #658): on some Apple-Silicon Macs an analysis task dies with

    RuntimeError: Could not parse python long as longdouble: 1 (Invalid argument)

raised while SciPy/scikit-learn is *imported* -- ``scipy/stats/_ksstats.py``
builds ``np.longdouble`` constants at import time, and ``tasks/paged_ivf.py``
imports ``sklearn.cluster`` to build the similarity search index.

Mechanism: NumPy converts a Python int to ``np.longdouble`` by formatting it to
a string and parsing it back through the C library. On macOS that path
(``NumPyOS_ascii_strtold``, compiled because ``HAVE_STRTOLD_L`` is defined)
runs, on EVERY conversion::

    newlocale(LC_ALL_MASK, "C", NULL)

and, when ``newlocale`` returns NULL, it leaves ``errno`` set to EINVAL; NumPy
then raises the error above. macOS' ``newlocale``/``setlocale`` are not safe to
run concurrently, so when an Apple framework (Cocoa/CoreAudio, pulled in via
pyobjc/onnxruntime/PyAV) changes the process locale on another thread at the
instant of the conversion, ``newlocale`` transiently fails. It is a race, which
is why other projects hit it only intermittently and "fix" it by retrying.

Why pinning LC_NUMERIC is NOT the fix: NumPy builds its own private "C" locale
via ``newlocale`` and does NOT consult the process/global locale for this call,
so ``setlocale(LC_NUMERIC, "C")`` and an ``LC_NUMERIC=C`` env var (kept as cheap
locale-churn hygiene) cannot stop the failing ``newlocale``. Upgrading NumPy
does not help either: the code path is unchanged through current releases.

The actual fix (``warmup_scipy_longdouble``): import ``scipy.stats`` and
``sklearn.cluster`` once, early, in the freshly started worker/Flask process --
a single-threaded moment before any Apple framework thread exists to race the
locale -- retrying the documented transient failure. The modules are then
cached, so:

* the RQ workers fork a child per job and the child INHERITS the already
  imported modules, so the index-build import in the child is a no-op and never
  re-runs the fragile conversion;
* the waitress/Flask request threads likewise reuse the cached import.

macOS-only: imported solely by ``native-build/macos/launcher.py``. The
Linux/Docker and Windows builds are unaffected -- glibc's built-in C locale
never fails and there is no Cocoa/CoreAudio racing the locale, while the MSVC
NumPy build compiles the locale-independent fallback that never calls
``newlocale`` -- so they neither import this module nor need it.
"""
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def pin_numeric_locale():
    """Pin LC_NUMERIC to C process-wide (best-effort locale-churn hygiene)."""
    os.environ["LC_NUMERIC"] = "C"
    try:
        import locale
        locale.setlocale(locale.LC_NUMERIC, "C")
    except Exception:
        pass


def warmup_scipy_longdouble(attempts=8, delay=0.1):
    """Import the longdouble-fragile modules once, retrying the macOS race.

    Returns True once ``np.longdouble`` conversion plus ``scipy.stats`` and
    ``sklearn.cluster`` import cleanly (or were already imported), False if every
    attempt failed. Best-effort: never raises, so it can never block startup. A
    genuinely missing module is not the transient race, so it short-circuits the
    retries; on permanent failure a warning is logged with the last error so a
    broken install is diagnosable instead of starting up silently.
    """
    import importlib
    last_error = None
    for _ in range(max(1, int(attempts))):
        try:
            import numpy
            numpy.longdouble(1)
            importlib.import_module("scipy.stats")
            importlib.import_module("sklearn.cluster")
            return True
        except ModuleNotFoundError as exc:
            last_error = exc
            break
        except Exception as exc:
            last_error = exc
            for _name in ("sklearn.cluster", "scipy.stats"):
                sys.modules.pop(_name, None)
            time.sleep(delay)
    logger.warning(
        "warmup_scipy_longdouble gave up importing scipy.stats/sklearn.cluster "
        "(last error: %r); analysis tasks may fail when they first use these modules.",
        last_error,
    )
    return False
