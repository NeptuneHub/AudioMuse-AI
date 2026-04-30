import os
import logging


logger = logging.getLogger(__name__)


def get_env(key, default=""):
    """Read an environment variable, preferring KEY_FILE when present."""
    file_path = os.environ.get(f"{key}_FILE")
    if file_path:
        try:
            with open(file_path, encoding="utf-8") as file_handle:
                return file_handle.read().strip()
        except OSError as exc:
            logger.warning("Failed to read %s_FILE from %s: %s", key, file_path, exc)
    return os.environ.get(key, default)
