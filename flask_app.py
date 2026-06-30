import os
import sys

from flask import Flask


def _resource_root():
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(os.path.abspath(__file__))


_RESOURCE_ROOT = _resource_root()
app = Flask(
    __name__,
    template_folder=os.path.join(_RESOURCE_ROOT, 'templates'),
    static_folder=os.path.join(_RESOURCE_ROOT, 'static'),
)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024
