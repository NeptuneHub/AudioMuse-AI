"""Flask application instance.

This module exists solely to host the singleton ``Flask(__name__)`` used by
both the web layer (``app.py``) and the RQ task modules under ``tasks/``.
Keeping the instance here avoids the import cycle that occurs when ``app.py``
imports task modules and a task module imports ``app`` to grab the Flask
instance for ``app.app_context()``.

Both sides should ``from flask_app import app``; ``app.py`` is then free to
attach blueprints, hooks, and middleware without creating a cycle.
"""

from flask import Flask

app = Flask(__name__)
# Cap upload size at 5GB; chunked backup uploads (see templates/backup.html)
# stay well under this, but allow some headroom for larger single requests.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
