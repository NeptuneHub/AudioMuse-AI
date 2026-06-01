from error.error_dictionary import (
    ERROR_REGISTRY,
    UNKNOWN_ERROR_CODE,
    get_error_class,
    get_default_message,
)
from error.error_manager import (
    AudioMuseError,
    ErrorManager,
    build,
    record,
    classify,
    from_exception,
    http_status_for_code,
)

__all__ = [
    "ERROR_REGISTRY",
    "UNKNOWN_ERROR_CODE",
    "get_error_class",
    "get_default_message",
    "AudioMuseError",
    "ErrorManager",
    "build",
    "record",
    "classify",
    "from_exception",
    "http_status_for_code",
]
