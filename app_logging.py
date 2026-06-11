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

Emoji safety: the ``EmojiStrippingFilter`` removes emoji and other non-Latin-1
symbols from every log record before it reaches a handler.  This prevents
``UnicodeEncodeError`` / ``UnicodeDecodeError`` crashes on Windows when stdout
is a pipe (PyInstaller native build) or when the console code-page cannot
represent the character.  HTML templates and web-UI progress messages are
unaffected — only the Python ``logging`` pipeline is sanitised.
"""

import logging
import re

LOG_FORMAT = "[%(levelname)s]-[%(asctime)s]-%(message)s"
LOG_DATEFMT = "%d-%m-%Y %H-%M-%S"

# ---------------------------------------------------------------------------
# Emoji / symbol stripping for console-safe logging
# ---------------------------------------------------------------------------
# Ranges cover all common emoji blocks plus Dingbats, Misc Symbols,
# Geometric Shapes, Supplemental Symbols, and variation selectors.
# Characters within Latin-1 (U+0000–U+00FF) are *not* stripped, so
# European accented letters (e.g. é, ñ, ü) pass through unchanged.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF"   # Misc Symbols, Emoticons, Transport, Supplemental
    "\U0001FA00-\U0001FAFF"    # Chess Symbols, Symbols Extended-A
    "\U00002190-\U000027BF"    # Arrows (→ ← ↑ ↓ ↔), Misc Technical, Dingbats (✓ ✗ ✕ ★ ☆ ♯ ♭ etc.)
    "\U000025A0-\U000025FF"    # Geometric Shapes (● ○ ■ □ ◆ ◇ ▲ ▼ etc.)
    "\U00002B00-\U00002BFF"    # Misc Symbols & Arrows
    "\U0001F000-\U0001F02F"    # Mahjong Tiles
    "\U0001F0A0-\U0001F0FF"    # Playing Cards
    "\uFE0F\u200D"             # Variation Selector-16, Zero-Width Joiner
    "]+"
)


def _strip_emoji(text: str) -> str:
    """Remove emoji and symbol characters from *text*, returning a plain string."""
    if not isinstance(text, str):
        return text
    cleaned = _EMOJI_RE.sub("", text)
    # Collapse multiple spaces that may result from removing a symbol
    return re.sub(r" {2,}", " ", cleaned).strip()


class EmojiStrippingFilter(logging.Filter):
    """Logging filter that strips emoji/symbols from ``record.msg`` and ``record.args``.

    Attach this to the root logger's handlers so every log record is sanitised
    before it reaches a ``StreamHandler`` (console / pipe).  File-based handlers
    with ``propagate=False`` (e.g. the Windows supervisor's own log) are not
    affected.
    """

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = _strip_emoji(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _strip_emoji(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    _strip_emoji(a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Install the project-wide root logger format. Idempotent.

    An ``EmojiStrippingFilter`` is attached to every handler on the root logger,
    making all console / pipe output safe on Windows regardless of code-page.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    for handler in logging.root.handlers:
        if not any(isinstance(f, EmojiStrippingFilter) for f in handler.filters):
            handler.addFilter(EmojiStrippingFilter())
