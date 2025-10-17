from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.song_alchemy import song_alchemy
import config

logger = logging.getLogger(__name__)

alchemy_bp = Blueprint('alchemy_bp', __name__, template_folder='../templates')


@alchemy_bp.route('/alchemy', methods=['GET'])
def alchemy_page():
    return render_template('alchemy.html')


@alchemy_bp.route('/api/alchemy', methods=['POST'])
def alchemy_api():
    """POST payload: {"add": [{"id":"...","op":"ADD"}, ...], "n":100}
    Expect at least two ADD items and up to 10 per group.
    """
    payload = request.get_json() or {}
    items = payload.get('items', [])
    n = payload.get('n', config.ALCHEMY_DEFAULT_N_RESULTS)

    add_ids = [i['id'] for i in items if i.get('op', '').upper() == 'ADD']
    sub_ids = [i['id'] for i in items if i.get('op', '').upper() == 'SUBTRACT']

    # Allow optional override for subtract distance (from frontend slider)
    subtract_distance = payload.get('subtract_distance')
    try:
        results = song_alchemy(add_ids, sub_ids, n_results=n, subtract_distance=subtract_distance)
        # song_alchemy now returns a dict with results, filtered_out and centroid projections
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Alchemy failure")
        return jsonify({"error": "Internal error"}), 500
