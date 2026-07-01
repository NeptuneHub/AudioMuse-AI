# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared logging setup for every AudioMuse-AI entry point.

Call ``configure_logging()`` exactly once per process, as early as possible,
before any module that emits log records is imported. Workers that don't import
``app`` (the high-priority worker, the janitor) call this instead of setting up
logging inline, so formats stay uniform and no ``logger.info`` silently falls
through to Python's ``lastResort`` handler during long jobs.

Main Features:
* One shared format and a re-entrant setup (``basicConfig`` is a no-op once the
  root logger has a handler), safe to call from any entry point.
* ``LogSanitizingFilter`` cleans every record before a handler sees it: strips
  emoji / non-Latin-1 symbols that raise UnicodeEncodeError on Windows pipes,
  and neutralizes CR/LF and other control chars to block log injection
  (CWE-117), while leaving ``logger.exception`` tracebacks intact.
"""

import logging
import re
from typing import Any

LOG_FORMAT = "[%(levelname)s]-[%(asctime)s]-%(message)s"
LOG_DATEFMT = "%d-%m-%Y %H-%M-%S"

# ---------------------------------------------------------------------------
# Console-safe + injection-safe log record sanitization
# ---------------------------------------------------------------------------
# Ranges cover all common emoji blocks plus Dingbats, Misc Symbols,
# Geometric Shapes, Supplemental Symbols, and variation selectors.
# Characters within Latin-1 (U+0000-U+00FF) are *not* stripped, so
# European accented letters pass through unchanged.
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001f9ff"  # Misc Symbols, Emoticons, Transport, Supplemental
    "\U0001fa00-\U0001faff"  # Chess Symbols, Symbols Extended-A
    "\U00002190-\U000027bf"  # Arrows, Misc Technical, Dingbats
    "\U000025a0-\U000025ff"  # Geometric Shapes
    "\U00002b00-\U00002bff"  # Misc Symbols & Arrows
    "\U0001f000-\U0001f02f"  # Mahjong Tiles
    "\U0001f0a0-\U0001f0ff"  # Playing Cards
    "\\uFE0F\\u200D"  # Variation Selector-16, Zero-Width Joiner
    "]+"
)

# Control codes plus DEL and the Unicode line/paragraph separators, EXCEPT tab
# (0x09). Includes LF (0x0A), CR (0x0D), NEL (U+0085), LINE SEPARATOR (U+2028)
# and PARAGRAPH SEPARATOR (U+2029): replacing these with a space prevents a
# value with an embedded line break from forging additional log lines, including
# for consumers (and Windows code-pages) that treat the Unicode separators as
# line boundaries.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0A-\x1F\x7F\x85" + chr(0x2028) + chr(0x2029) + "]")


def _sanitize_log_text(text: Any) -> Any:
    """Make *text* safe for a single console log line.

    Strips emoji/symbol characters (Windows code-page safety) and replaces CR/LF
    and other C0 control codes with a space so an attacker-controlled value
    cannot forge or split log lines (CWE-117). Tabs are preserved. Non-string
    values (e.g. numeric ``logging`` args) are returned unchanged.
    """
    if not isinstance(text, str):
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    # Collapse runs of spaces left behind by removed symbols / control codes.
    return re.sub(r" {2,}", " ", cleaned).strip()


class LogSanitizingFilter(logging.Filter):
    """Logging filter that sanitizes ``record.msg`` and ``record.args``.

    Attach this to the root logger's handlers so every log record is cleaned
    before it reaches a ``StreamHandler`` (console / pipe): emoji/symbols are
    removed and CR/LF + control characters are neutralised. File-based handlers
    with ``propagate=False`` (e.g. the Windows supervisor's own log) are not
    affected. The exception traceback added by ``logger.exception`` lives in
    ``record.exc_info`` and is rendered by the formatter after this filter runs,
    so it is never altered here - the full error always reaches the log.
    """

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = _sanitize_log_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _sanitize_log_text(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    _sanitize_log_text(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Install the project-wide root logger format. Idempotent.

    A ``LogSanitizingFilter`` is attached to every handler on the root logger,
    making all console / pipe output safe on Windows regardless of code-page and
    neutralising log-injection attempts from untrusted message data.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    for handler in logging.root.handlers:
        if not any(isinstance(f, LogSanitizingFilter) for f in handler.filters):
            handler.addFilter(LogSanitizingFilter())
