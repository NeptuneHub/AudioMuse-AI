import locale
import os
from unittest.mock import patch

import numeric_bootstrap


def test_pin_numeric_locale_sets_env_and_calls_setlocale(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "en_US.UTF-8")
    with patch("locale.setlocale") as mock_setlocale:
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"
    mock_setlocale.assert_called_once_with(locale.LC_NUMERIC, "C")


def test_pin_numeric_locale_setlocale_error_does_not_propagate(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "en_US.UTF-8")
    with patch("locale.setlocale", side_effect=locale.Error("unsupported locale")) as mock_setlocale:
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"
    mock_setlocale.assert_called_once_with(locale.LC_NUMERIC, "C")


def test_pin_numeric_locale_generic_exception_still_sets_env_when_unset(monkeypatch):
    monkeypatch.setenv("LC_NUMERIC", "placeholder")
    monkeypatch.delenv("LC_NUMERIC")
    with patch("locale.setlocale", side_effect=Exception("boom")):
        numeric_bootstrap.pin_numeric_locale()
    assert os.environ["LC_NUMERIC"] == "C"
