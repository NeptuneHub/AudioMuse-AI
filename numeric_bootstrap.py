import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def pin_numeric_locale():
    os.environ["LC_NUMERIC"] = "C"
    try:
        import locale
        locale.setlocale(locale.LC_NUMERIC, "C")
    except Exception:
        pass


def warmup_scipy_longdouble(attempts=8, delay=0.1):
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
