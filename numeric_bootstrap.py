"""Pin a deterministic C numeric locale for the standalone macOS app.

Works around an intermittent macOS-arm64 NumPy bug: converting a Python int to
``np.longdouble`` goes through the C library's locale-sensitive ``strtold``, and
when an Apple framework (Cocoa / CoreAudio, pulled in via pyobjc / onnxruntime)
changes the process locale at startup, the conversion can fail with::

    RuntimeError: Could not parse python long as longdouble: 1 (Invalid argument)

SciPy performs that conversion at *import* time (``scipy/stats/_ksstats.py``), so
on the affected machines importing scipy/sklearn intermittently crashes the
analysis worker and leaves the task stuck pending.

Pinning ``LC_NUMERIC`` to ``"C"`` makes ``strtold`` deterministic and removes the
locale transition there is to race against. It is kept narrow -- only the numeric
category, so ``LC_CTYPE``/UTF-8 (accented file paths) is untouched.

This module is imported ONLY by the macOS launcher (``macos/launcher.py``); it is
never referenced by the Linux/Docker worker entrypoints, so the container images
are entirely unaffected. The matching env-level pin lives in ``macos/env.py``.
"""
import os


def pin_numeric_locale():
    """Force a deterministic C numeric locale (process-wide), best-effort."""
    os.environ["LC_NUMERIC"] = "C"
    try:
        import locale
        locale.setlocale(locale.LC_NUMERIC, "C")
    except Exception:
        # Never let locale setup abort process startup.
        pass
