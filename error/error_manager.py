"""Central error manager for AudioMuse-AI.

The rest of the codebase calls this module to turn a chosen error code (and an
optional one-line detail) into the canonical structured error that the frontend
renders:

    {"error_code": int, "error_class": str, "error_message": str}

The user-facing ``error_message`` is always collapsed to a single line and never
carries a stack trace. The full traceback is sent to the logger; it is added to
the returned dict only when the centralized ``ERROR_INCLUDE_TRACEBACK`` toggle is
enabled, so debugging stays opt-in.
"""
import logging
import traceback

from config import ERROR_INCLUDE_TRACEBACK
from error.error_dictionary import (
    ERROR_REGISTRY,
    UNKNOWN_ERROR_CODE,
    ERR_MEDIASERVER_REFUSED,
    ERR_MEDIASERVER_TIMEOUT,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_DB_CONNECTION,
    ERR_INDEX_EMPTY,
    get_error_class,
    get_default_message,
)

logger = logging.getLogger(__name__)

_MAX_MESSAGE_DETAIL = 400

_EXCEPTION_NAME_CODES = {
    "ConnectionError": ERR_MEDIASERVER_REFUSED,
    "ConnectionRefusedError": ERR_MEDIASERVER_REFUSED,
    "NewConnectionError": ERR_MEDIASERVER_REFUSED,
    "MaxRetryError": ERR_MEDIASERVER_UNREACHABLE,
    "HTTPError": ERR_MEDIASERVER_UNREACHABLE,
    "ConnectTimeout": ERR_MEDIASERVER_TIMEOUT,
    "ConnectTimeoutError": ERR_MEDIASERVER_TIMEOUT,
    "ReadTimeout": ERR_MEDIASERVER_TIMEOUT,
    "ReadTimeoutError": ERR_MEDIASERVER_TIMEOUT,
    "Timeout": ERR_MEDIASERVER_TIMEOUT,
    "timeout": ERR_MEDIASERVER_TIMEOUT,
    "OperationalError": ERR_DB_CONNECTION,
    "LyrionAPIError": ERR_MEDIASERVER_UNREACHABLE,
    "EmptyIndexError": ERR_INDEX_EMPTY,
}


def _one_line(text):
    return " ".join(str(text).split())


def build(code, message=None):
    """Return the canonical {error_code, error_class, error_message} dict."""
    resolved_code = code if code in ERROR_REGISTRY else UNKNOWN_ERROR_CODE
    error_class = get_error_class(resolved_code)
    base = get_default_message(resolved_code)
    detail = _one_line(message) if message else ""
    if detail and len(detail) > _MAX_MESSAGE_DETAIL:
        detail = detail[: _MAX_MESSAGE_DETAIL - 3].rstrip() + "..."
    full = f"{base} {detail}".strip() if detail else base
    return {"error_code": resolved_code, "error_class": error_class, "error_message": full}


def classify(exc, default_code=UNKNOWN_ERROR_CODE):
    """Map an exception to a registry code by type, falling back to default_code."""
    if isinstance(exc, AudioMuseError):
        return exc.code
    return _EXCEPTION_NAME_CODES.get(type(exc).__name__, default_code)


def http_status_for_code(code):
    """Return the HTTP status a synchronous handler should use for a code."""
    if 1100 <= code < 1200:
        return 502
    if 1000 <= code < 1100:
        return 400
    if 4000 <= code < 4100:
        return 503
    return 500


class AudioMuseError(Exception):
    """Raised by synchronous code paths that want a structured, coded error.

    ``str(err)`` yields the single-line user-facing message, so it is always safe
    to display directly. ``cause`` is kept for logging only.
    """

    def __init__(self, code, message=None, cause=None):
        self.code = code if code in ERROR_REGISTRY else UNKNOWN_ERROR_CODE
        self.error_class = get_error_class(self.code)
        built = build(self.code, message)
        self.error_message = built["error_message"]
        self.cause = cause
        super().__init__(self.error_message)

    def to_dict(self):
        return {
            "error_code": self.code,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }

    def __str__(self):
        return self.error_message


def record(code, message=None, exc=None, logger=None, level=logging.ERROR):
    """Build the structured error and, when a logger is given, log full detail.

    Returns the canonical dict. The full traceback is attached to the dict only
    when ERROR_INCLUDE_TRACEBACK is enabled.
    """
    err = build(code, message)
    if logger is not None:
        logger.log(
            level,
            "[%s] %s: %s",
            err["error_code"],
            err["error_class"],
            err["error_message"],
            exc_info=exc if exc is not None else False,
        )
    if ERROR_INCLUDE_TRACEBACK and exc is not None:
        err["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return err


def from_exception(exc, code=None, message=None, logger=None, level=logging.ERROR):
    """Build a structured error from any exception.

    An ``AudioMuseError`` keeps its own code/class/message. Any other exception is
    classified by type (or uses ``code`` when supplied) and its ``str()`` becomes
    the one-line detail.
    """
    if isinstance(exc, AudioMuseError):
        err = exc.to_dict()
        if logger is not None:
            logger.log(
                level,
                "[%s] %s: %s",
                err["error_code"],
                err["error_class"],
                err["error_message"],
                exc_info=exc,
            )
        if ERROR_INCLUDE_TRACEBACK:
            err["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        return err
    resolved = code if code is not None else classify(exc, UNKNOWN_ERROR_CODE)
    detail = message if message is not None else str(exc)
    return record(resolved, detail, exc=exc, logger=logger, level=level)


class ErrorManager:
    """Convenience facade grouping the module-level helpers."""

    AudioMuseError = AudioMuseError
    build = staticmethod(build)
    record = staticmethod(record)
    classify = staticmethod(classify)
    from_exception = staticmethod(from_exception)
    http_status_for_code = staticmethod(http_status_for_code)
