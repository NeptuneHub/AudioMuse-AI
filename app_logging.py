"""Shared logging setup for every AudioMuse-AI entry point.

Call ``configure_logging()`` exactly once per process, as early as possible —
before any module that emits log records gets imported. ``logging.basicConfig``
is a no-op if the root logger already has a handler, so calling this helper
multiple times across imports is safe.

Why this exists: workers that don't import ``app`` (the high-priority worker,
the janitor) used to set up logging inline, with formats that drifted from
``app.py``. When one of them forgot to call ``basicConfig`` at all, every
``logger.info(...)`` from task modules fell through to Python's ``lastResort``
handler — silently dropping INFO-level output during long-running jobs.

Record sanitization: the ``LogSanitizingFilter`` cleans every log record before
it reaches a handler. It removes emoji / non-Latin-1 symbols (which raise
``UnicodeEncodeError`` / ``UnicodeDecodeError`` on Windows when stdout is a pipe
or the console code-page cannot represent the character) and neutralizes CR/LF
and other control characters so an attacker-controlled value embedded in a
message cannot forge or split log lines (CWE-117, log injection). This is the
single, centralized place where log-message sanitization happens — call sites
log the raw value and the filter cleans it. HTML templates and web-UI progress
messages are unaffected — only the Python ``logging`` pipeline is sanitised, and
the traceback ``logger.exception`` appends is left intact (the formatter renders
it after filtering, so the full error is always visible in the log).
"""

import logging
import re

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
    "[\U0001F300-\U0001F9FF"   # Misc Symbols, Emoticons, Transport, Supplemental
    "\U0001FA00-\U0001FAFF"    # Chess Symbols, Symbols Extended-A
    "\U00002190-\U000027BF"    # Arrows, Misc Technical, Dingbats
    "\U000025A0-\U000025FF"    # Geometric Shapes
    "\U00002B00-\U00002BFF"    # Misc Symbols & Arrows
    "\U0001F000-\U0001F02F"    # Mahjong Tiles
    "\U0001F0A0-\U0001F0FF"    # Playing Cards
    "\\uFE0F\\u200D"           # Variation Selector-16, Zero-Width Joiner
    "]+"
)

# C0 control codes plus DEL, EXCEPT tab (0x09). Includes LF (0x0A) and CR (0x0D):
# replacing these with a space prevents a value with embedded newlines from
# forging additional log lines.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0A-\x1F\x7F]")


def _sanitize_log_text(text: str) -> str:
    """Make *text* safe for a single console log line.

    Strips emoji/symbol characters (Windows code-page safety) and replaces CR/LF
    and other C0 control codes with a space so an attacker-controlled value
    cannot forge or split log lines (CWE-117). Tabs are preserved.
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
    so it is never altered here — the full error always reaches the log.
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
                    _sanitize_log_text(a) if isinstance(a, str) else a
                    for a in record.args
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
