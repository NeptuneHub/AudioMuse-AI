"""Flask application instance.

This module exists solely to host the singleton ``Flask(__name__)`` used by
both the web layer (``app.py``) and the RQ task modules under ``tasks/``.
Keeping the instance here avoids the import cycle that occurs when ``app.py``
imports task modules and a task module imports ``app`` to grab the Flask
instance for ``app.app_context()``.

Both sides should ``from flask_app import app``; ``app.py`` is then free to
attach blueprints, hooks, and middleware without creating a cycle.
"""

import os
import sys

from flask import Flask


def _resource_root():
    """Root that holds ``templates/`` and ``static/`` (the PyInstaller bundle when frozen)."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(os.path.abspath(__file__))


_RESOURCE_ROOT = _resource_root()
app = Flask(
    __name__,
    template_folder=os.path.join(_RESOURCE_ROOT, 'templates'),
    static_folder=os.path.join(_RESOURCE_ROOT, 'static'),
)
# Cap upload size at 5GB; chunked backup uploads (see templates/backup.html)
# stay well under this, but allow some headroom for larger single requests.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
