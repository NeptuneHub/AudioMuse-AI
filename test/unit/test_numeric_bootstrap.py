import locale
import os
import sys
from unittest.mock import patch

import pytest

import numeric_bootstrap


@pytest.fixture
def _preserve_fragile_modules():
    saved = {name: sys.modules.get(name) for name in ("scipy.stats", "sklearn.cluster")}
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_pin_numeric_locale_sets_env_and_calls_setlocale(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "en_US.UTF-8")
    with patch("locale.setlocale") as mock_setlocale:
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"
    mock_setlocale.assert_called_once_with(locale.LC_NUMERIC, "C")


def test_pin_numeric_locale_setlocale_error_does_not_propagate(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "en_US.UTF-8")
    with patch(
        "locale.setlocale", side_effect=locale.Error("unsupported locale")
    ) as mock_setlocale:
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"
    mock_setlocale.assert_called_once_with(locale.LC_NUMERIC, "C")


def test_pin_numeric_locale_generic_exception_still_sets_env_when_unset(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "placeholder")
    monkeypatch.delenv("LC_NUMERIC")
    with patch("locale.setlocale", side_effect=Exception("boom")):
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"


def test_warmup_imports_and_caches_fragile_modules():
    import sys

    assert numeric_bootstrap.warmup_scipy_longdouble() is True
    assert "scipy.stats" in sys.modules
    assert "sklearn.cluster" in sys.modules


def test_warmup_retries_transient_longdouble_failure(monkeypatch, _preserve_fragile_modules):
    import numpy

    calls = {"n": 0}
    real = numpy.longdouble

    def flaky(value):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Could not parse python long as longdouble: 1 (Invalid argument)")
        return real(value)

    monkeypatch.setattr(numpy, "longdouble", flaky)
    assert numeric_bootstrap.warmup_scipy_longdouble(attempts=5, delay=0) is True
    assert calls["n"] == 3


def test_warmup_gives_up_and_returns_false_without_raising(monkeypatch, _preserve_fragile_modules):
    import numpy

    def always_fail(value):
        raise RuntimeError("boom")

    monkeypatch.setattr(numpy, "longdouble", always_fail)
    assert numeric_bootstrap.warmup_scipy_longdouble(attempts=3, delay=0) is False
