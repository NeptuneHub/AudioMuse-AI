import logging

from flask import Blueprint, render_template

logger = logging.getLogger(__name__)

users_bp = Blueprint('users_bp', __name__)


@users_bp.route('/users')
def users_page():
    """
    Users admin page.
    ---
    tags:
      - Users
    summary: HTML page for managing AudioMuse users.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('users.html', title='AudioMuse-AI - Users', active='users')
