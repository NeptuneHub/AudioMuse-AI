# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Numeric-stack startup guards for scipy/sklearn and numpy longdouble.

Provides bootstrap helpers imported before the RQ worker forks, working around
the macOS newlocale race that breaks ``numpy.longdouble`` parsing and warming
the scipy/sklearn imports so analysis tasks do not fail on first use.

Main Features:
* ``warmup_scipy_longdouble`` retries the fragile imports and reports failure once.
* ``pin_numeric_locale`` forces the C numeric locale (a no-op on some platforms).
"""

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
