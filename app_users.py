import logging

from flask import Blueprint, render_template

logger = logging.getLogger(__name__)

users_bp = Blueprint('users_bp', __name__)


@users_bp.route('/users')
def users_page():
    return render_template('users.html', title='AudioMuse-AI - Users', active='users')
